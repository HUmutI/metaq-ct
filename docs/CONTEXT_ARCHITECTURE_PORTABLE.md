# Indication + age + sex conditioning for a Q-Former CT model

A description of the mechanism, written to be ported to another codebase. It
assumes a CLIP-style 3D CT model whose image tower ends in a Q-Former: a bank of
learnable query vectors that cross-attend to the feature map and produce one token
per query, pooled into the image embedding.

The design constraint everything else follows from:

> The indication does not tell the model *what is present*. It tells the model
> *where and what to interrogate more carefully*, while an indication-independent
> representation preserves comprehensive interpretation of the scan.

That sentence is why the model keeps two read-outs, not one, and why the
unconditioned one must remain provably untouched.

---

## 0. The one invariant

**At initialisation the conditioned model must be numerically identical to the
model you started from** — same output, to the last bit, for every input. Every
new module is zero-initialised or zero-gated so that its contribution at step 0 is
exactly zero.

This is worth the trouble because it turns every later measurement into "how far
did we move from the baseline", rather than "these are two different models that
happen to be related". Without it you cannot attribute a gain, and you cannot tell
a bug from an effect.

Make it a test, not an intention. Ours asserts, in fp32 with deterministic kernels:

- `Z_final == Z_gen` exactly (not "close"),
- the unconditioned query tokens match the original model's to `0.0e+00`,
- `Z_gen` is unchanged when the indication is swapped, **with the gates forced
  open** — this is the isolation claim itself, and it catches a self-attention or
  pooling regression at any point in training, not just at step 0,
- no gradient path runs from `Z_gen` into any context module.

---

## 1. The query bank

Start from the original bank. Ours is `[10 anatomy | C pathology | 2 global]`,
where `C` is the number of classes. Widen the **attention bank** while adding
almost nothing to the **parameter tensor**:

```
slot   0 .. 9        anatomy                 -> param rows 0..9     (unchanged)
slot  10 .. 9+C      pathology, GENERAL      -> param rows 10..9+C  (unchanged)
slot     .. +2       global,   GENERAL       -> param rows ...      (unchanged)
--------------------------------------- n_gen boundary ------------------
slot     .. +C       pathology, CONDITIONED  -> param rows 10..9+C  (WEIGHT-TIED)
slot     last        global,   CLINICAL      -> ONE new parameter row
```

For `C = 18` that is **49 attention slots over 31 parameter rows**; for `C = 27`,
67 slots over 40 rows. The conditioned pathology block is not new parameters — it
is the general block *read a second time*, so the two twins start from the same
query vector and differ only in what they are allowed to attend to and in whether
they were conditioned.

Four things depend on unconditioned-first-and-contiguous:

1. `queries[:n_gen]` stays byte-identical to the original tensor, so a pretrained
   checkpoint loads by **appending one row**, with no remapping.
2. `gen` and `cond` are contiguous slices, so the group constraint in §4 is two
   attention calls rather than a large additive mask.
3. The original role/anatomy mask keeps its meaning for rows `0..n_gen-1`;
   extending it is one `index_select`.
4. `Z_gen` is just `tokens[:, :n_gen].mean(1)`.

**Weight tying is one `index_select`, not parameter aliasing.**
`queries.index_select(0, slot_to_row)` emits rows `10..9+C` twice; the backward of
`index_select` is `index_add`, so gradients from both twins sum into the same row
automatically. Aliasing two `nn.Parameter`s instead makes `state_dict()` write two
keys and one silently overwrites the other on load.

**Keep two index vectors, not one.** `slot_to_row` says which *parameter row* a
slot reads. `role_row` says which *attention-mask row* it inherits. They agree
everywhere except the clinical global slot, which uses the new parameter row but
inherits an unrestricted global mask row. Conflating them is the easiest mistake
in the whole change.

---

## 2. The context tokens `H_C`

A short sequence the conditioned queries attend to:

```
H_C = [ indication tokens (L) ; age ; sex ; age×sex ]      -> (B, L+3, D)
```

**Indication.** Free text through a **frozen** copy of the same text tower the
model already uses for reports, then LayerNorm. Frozen for two reasons: the
indication is an input, not a target, and the relevance head in §5 compares
indication embeddings against class-prompt embeddings — both must live in one
stable basis or the comparison drifts as the image tower trains.

Keep the frozen tower **out of `state_dict()`** (hold it in a plain list attribute
and override `_apply`/`train`), or every checkpoint carries a second copy of BERT
and `load_state_dict(strict=True)` starts arguing about it.

Truncate to ~64 tokens. Indications are short; the tail is boilerplate.

**Missing indication is a legitimate input, not a hole.** Use one literal
`NO_INDICATION` string so the model sees a consistent token sequence rather than
an empty one. In our adult cohort only 49% of volumes carry an indication at all,
and the model has to work on the other half.

**Age.** For a multi-age cohort we use an **ordinal-cumulative** embedding over
bands, so that "older than" is built into the parameterisation:

```
e_age(b) = e_0 + sum_{j<b} softplus(delta_j)
```

with a **separate free row** for unknown age, outside the cumulative chain —
folding unknowns into band 0 would make every missing age a newborn.

> **For a CT-RATE-only project this is over-engineered.** CT-RATE is adults: in
> our split six of the ten bands are empty and always will be. The cumulative sum
> then adds the same untrained constant to every band you actually use, which the
> base embedding simply absorbs. Use a plain embedding table over the 4–5 bands
> that occur (or a single normalised scalar plus an unknown flag) and drop the
> ordinal machinery. Keep the unknown row either way.
>
> One trap if you do keep bands: a band that is empty in TRAIN but present in
> VALID is a real fault — the model is asked at evaluation for an increment it
> never learned. A band empty in both is just a property of the cohort. Check for
> the first, not the second. And watch thin bands: ours had a band with seven
> training volumes whose increment is shared by every band above it.

Init the increments small: `softplus(0) = 0.693` means `delta = 0` already gives a
large ramp across the bands. We use `delta_init = -3.0`, an increment of ~0.049.

**Sex** is a small embedding table with an explicit unknown row. **The interaction
token** is an embedding of `(age_band, sex)` initialised to exactly zero, so it
contributes nothing until it earns its place.

Age, sex and interaction are always three real tokens. **Dropout must never mask
them**, so `H_C` can never be entirely padding — a fully padded row makes
`MultiheadAttention` emit `NaN`, and a zero gate does not stop it, because
`0 * NaN = NaN`. That `NaN` reaches the final embedding and gets blamed on the
image tower.

---

## 3. C1 — conditioning the pathology queries

Applied **once**, to the conditioned query block, before the first Q-Former block:

```python
q_tilde = q + g_x * CrossAttn(LN(q), LN(H_C), LN(H_C))     # g_x scalar, init 0
c_bar   = masked_mean(LN(H_C))
gamma   = 1 + film_eps * tanh(W_gamma @ c_bar)             # W_gamma init 0
beta    = W_beta @ c_bar                                   # W_beta  init 0
q_out   = gamma * q_tilde + beta
```

Anatomy queries are **not** conditioned — anatomy determines where a finding can
occur, which the indication does not change.

Two details that are easy to get wrong:

- **Zero both the weight and the bias** of the FiLM projections. With only the
  weight zeroed, `beta` is a nonzero constant and the identity in §0 fails.
  `gamma = 1 + eps·tanh(0) = 1` and `beta = 0` gives exact identity.
- `film_eps` bounds the multiplicative effect (`gamma` in `1 ± eps`; we use 0.2).
  Without a bound FiLM can scale a query arbitrarily, which is a different and
  much less controllable mechanism than "modulate".

Once, not per block: the conditioning lives in the queries, and re-applying it at
every block compounds it in a way nothing in the design asked for.

---

## 4. Isolation — four paths, all four needed

`Z_gen` must be *provably* independent of the indication. Information can leak
back four ways, so all four are closed:

1. **Self-attention.** Within a Q-Former block, queries attend to each other, so a
   conditioned twin would leak into its general twin. Split it:

   ```python
   h_gen_out = SelfAttn(h_gen, h_gen, h_gen)   # general sees ONLY general
   h_con_out = SelfAttn(h_con, h_all, h_all)   # conditioned sees everything
   out = cat([h_gen_out, h_con_out], dim=1)
   ```

   Prefer this split over a `(S,S)` additive mask: the general half then runs the
   *same op with the same shape* as the original model, which keeps the step-0
   identity exact rather than "equal up to a different reduction order and a
   different SDPA kernel". Keep the mask version as a unit test — if the two
   disagree, one of your index vectors is wrong.

   Use `-inf` (a boolean mask), not a finite penalty. This is a structural
   invariant; a finite penalty leaks exactly the information isolation exists to
   block.

2. **Pooling.** `Z_gen` averages only the general slots.
3. **Fusion.** `W = [I | 0]` at init, so the conditioned half contributes nothing.
4. **Zero gates**, as in §3.

---

## 5. C2 — relevance-weighted pooling, which may raise but never suppress

Per class `c`, from the pooled indication embedding `e_ind` and the class-prompt
embedding `p_c` (both from the frozen tower):

```python
feats = cat([e_ind, p_c, e_ind * p_c], dim=-1)       # (B, C, 3D)
r     = sigmoid(MLP(feats))                          # (B, C), MLP: 3D -> 128 -> 1
beta  = softplus(b)                                  # scalar, >= 0 structurally
w     = 1 + beta * r                                 # (B, C), >= 1 structurally
```

**`w = 1 + beta·r`, never `w = r`.** A class the indication says nothing about
keeps weight 1; it is never scaled toward zero. Incidental findings — the thing a
radiologist most fears missing — are protected by the parameterisation rather than
by hoping the model learns not to suppress them. `beta = softplus(b)` makes
`w >= 1` a fact about the arithmetic, not something a clamp enforces.

**Normalise both the pool and the loss by `sum(w)`:**

```python
Z_pool = sum_c w_c * z_c / sum_c w_c
L_ind  = sum_c w_c * L_c / sum_c w_c
```

Without the normalisation `dL/d(beta) = sum_c r_c L_c > 0` always, and `beta` is
driven to its floor from the first step — the gate closes and never reopens. The
pool needs it too: an unnormalised `sum w_c z_c` grows with how many classes the
indication is relevant to, so a broad indication ("chest pain") produces a
systematically larger vector than a narrow one.

### The trap that cost us a whole ablation ladder

We initialised `b = -6`. At that point `d(beta)/db = sigmoid(-6) = 0.0025`, so the
gradient is effectively dead: over an entire run `beta` moved by a **measured**
0.00012 against a predicted 0.0001. The relevance head never switched on, and the
C2 rung was silently just C1 with an inert head bolted to it. Use `b = -2`.

**And the same class of mistake once more:** the context modules start at exactly
zero — that is what buys the step-0 identity — and they were given the same
learning rate as a warm-started pretrained ResNet. After 3,600 updates
`||W_ind|| / ||W_gen||` had reached 0.027, so the conditioned half carried under
3% of the general half, and three different rungs came out identical seed by seed
because the gates separating them never opened. Give the context parameters their
own optimizer group with a **~20× learning-rate multiplier**.

---

## 6. Read-out

```python
Z_gen   = tokens[:, :n_gen].mean(1)                       # unconditioned
Z_pool  = weighted_pool(tokens[:, cond_slots], w)         # C2
Z_clin  = tokens[:, clinical_slot]
Z_ind   = 0.5 * (Z_pool + Z_clin)                         # mean, not sum
Z_final = W @ cat([Z_gen, Z_ind])                         # W = [I | 0] at init
```

`Z_ind` is a **mean**, not a sum, so `||Z_ind||` stays on the scale of `||Z_gen||`
— otherwise the `||W_ind|| / ||W_gen||` diagnostic above is unreadable.

A gated sum `Z_final = LN(Z_gen + g(e_ind) * Z_ind)` is a reasonable alternative
arm. Note that with the gated arm the linear `W` is never used, so its diagnostic
is structurally zero and means nothing — do not read it as "the pathway is dead".

---

## 7. Losses

```
total = existing losses
      + w_gen * L_pertoken(general bank)        # unweighted, UNCHANGED
      + w_ind * L_pertoken(conditioned bank)    # weighted by w, normalised
      + w_cf  * L_counterfactual
```

Keep the original per-token loss on the general bank exactly as it was: it is what
holds the unconditioned pathway to its old behaviour.

If you have a region/organ alignment loss driven by report sentences, attach it to
the **general** bank. Wiring it to the conditioned bank teaches the exact opposite
of C1 — look where the *report* wrote, not where the *indication* points.

**Counterfactual consistency.** With probability ~0.25, re-run with another
volume's indication and penalise prediction changes on classes the relevance head
says are irrelevant:

```
L_cf = sum_c 1[r_c < tau] * || y_hat_c(I) - y_hat_c(I') ||^2
```

Two practicalities:

- **Reuse the image features.** Only `H_C` changed; re-running the visual forward
  doubles the step cost for nothing. Write that as a comment or someone will
  re-run it.
- **Reject the counterfactual partner on TEXT, not on identity.** Short
  requisitions repeat heavily — a partner drawn from a different volume frequently
  carries the identical string, and then `L_cf` compares a prediction with itself:
  exactly zero, no gradient, and the counterfactual signal diluted in proportion
  to how common the phrasing is.

---

## 8. Training-time details that are not optional

**Indication dropout ~0.3.** Drop only rows that *have* an indication; a row that
is already absent must not count as dropped, or with 49% coverage the effective
rate is ~0.85 and nothing in the log tells you.

**Derive the dropout draw from a seed, not `random.random()`.** It depends on
worker count and epoch order and is not restored by `torch.load`, and these jobs
requeue and auto-resume:

```python
rng = random.Random((ctx_seed * 1000003 + epoch * 7919 + idx) & 0x7FFFFFFF)
```

**`persistent_workers=False` when context is on.** If `self.epoch` never advances
in the worker, the dropout mask is frozen for the entire run: a fixed 30% of
volumes never show their indication once. That is a different — and worse —
experiment, and it looks exactly like the intended one.

**Dropout must be off during validation**, or checkpoint selection happens on 30%
blanked indications.

**Two gradient-clip groups, not one.** The new modules output zero at step 0 but
their *gradient* is not zero (`dL/dg_x = <a, dL/dq> != 0`). Under a single global
`clip_grad_norm_` their contribution inflates the total norm, so every inherited
parameter takes a smaller step than the reference run from update 1. The forward
identity survives that; the trajectory does not.

**Watch the RNG stream.** If `torch.manual_seed` fires at import, constructing any
new module *before* the Q-Former shifts the stream and silently changes
`queries` initialisation. Build the context modules **after** the existing model.

**Refresh cached class embeddings on resume.** The relevance head caches `p_c`; if
that is only refreshed on a periodic prompt re-encode, a resumed run scores
against a stale bank.

---

## 9. Evaluation

Four conditions, and the last two are where the claim is actually tested:

1. **no indication** — must match the baseline model. If it does not, isolation is
   broken.
2. **true indication** — the headline.
3. **shuffled indication** — another patient's. Should score between 1 and 2. If
   it scores like 2, the model is ignoring the indication; if far below 1, the
   conditioning is not a modulation but a dependency.
4. **mismatched indication**, adversarially chosen — the honest failure mode, and
   the one a reviewer will ask about.

Report the unconditioned read-out alongside the conditioned one. A model that only
works when handed a good indication is a different and weaker claim than one that
matches the baseline without and improves with.

---

## 10. Ablation order

Add one mechanism at a time, three seeds each, and **measure the seed band first**
— three seeds of an *identical* configuration. Ours is `sd = 0.0046`, so a delta
below ~0.0092 is not a difference. Without that number the whole ladder is
unreadable.

```
CT only  ->  + concat(indication)  ->  + cross-attn (C1, no FiLM)
         ->  + FiLM (C1 full)      ->  + C2
         ->  fusion {concat, gated}   isolation {split, off}   age {band, scalar, shuffled}
```

The `shuffled age` arm is the negative control: if it scores like `band`, age was
never being used.

Two reporting rules learned the hard way:

- **A quiet run directory is not a finished run.** A job killed at update 400
  leaves exactly what a finished run leaves — a best checkpoint, a metrics file,
  and silence. Nine of our first ladder's twenty-four tasks died that way, and
  averaging their early scores in made three arms look worse than they were.
  Treat a run below `patience x val_every` as dead and exclude it.
- **Validation AUC is a model-selection number, not a result.** It is computed on
  the split the run early-stopped against. Only held-out numbers are reportable,
  and only with a patient-clustered bootstrap interval beside them.
