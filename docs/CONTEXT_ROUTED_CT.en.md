<!--
  Context-Routed CT — architecture design document
  The pediatric + adult successor to ARC-CT · Kurugol Lab, BCH
  This file is TECHNICALLY IDENTICAL to its Turkish counterpart,
  CONTEXT_ROUTED_CT.tr.md. If a number or a decision changes in one,
  it must change in the other.
  Source artifact: claude.ai/code/artifact/7d8a60de-810f-4fa4-8868-524e30c7fffb
-->

`ARC-CT v2 · architecture design document · rev. 7`

# Context-Routed CT

Dr. Kurugol's name for it: **Indication-Conditioned Anatomy-Routed Attention**.
Anatomy determines *where* a finding can occur; the clinical indication determines
*which* findings deserve to be interrogated more carefully. And an
indication-independent pathway stays open at all times — because an unexpected
pneumothorax on a CT ordered for pneumonia cannot be missed.

The indication does not tell the model *what is present*. It tells the model
*where and what* to interrogate more carefully — while the indication-independent
representation preserves the comprehensive reading of the CT.

This document puts Dr. Kurugol's 25 KB of notes at the centre of the architecture:
**a dual pathway** (conditioned and unconditioned pathology representations together),
**cross-attention onto the indication token sequence**, the non-suppressing
**relevance gate** w = 1 + βr, and **4 free indication queries** that catch what falls
outside the 27 classes. On top of that, **four isolation fixes** (§04) that make the
unconditioned pathway genuinely unconditioned — without them the design's headline
safety claim was being violated silently. Measured numbers are marked `[measured]`,
design decisions `[proposal]`, open questions `[open]`.

> **rev.7 · scope — what Phase 1 delivers**
>
> **The novelty is C1 + C2.** Phase 1 delivers only these: the indication entering the
> pathology queries through cross-attention + FiLM (C1), non-suppressing relevance
> weighting (C2), isolation of the unconditioned pathway, and closing the empty-region
> cliff as a *bug fix*.
>
> **C3 (the learned λ gate) and C4 (free indication queries) are not in Phase 1.**
> Both remain as optional arms at the end of the ladder. Reasons below (§07, §08); in
> short: C3's motivation rests on a measurement that §03 Fact 2 shows is confounded,
> *and* through ρ it would require a mask at inference, risking ARC-CT's published
> mask-free-inference property; C4 is the largest shortcut surface in the design.
>
> This costs nothing and buys a lot: the ladder shrinks 12 → **8 rungs**, macro
> comparisons 36 → **24**, the seed budget ~675 → **~450 GPU-hours**, and the query
> bank 71 → **67 slots**. **And the paper is still complete:** if C1+C2 clear their own
> bars, the pediatric cohort is already a contribution on its own — there is no public
> pediatric 3D chest-CT VLM benchmark.
>
> **The phase split is attribution hygiene, not a budget constraint.** GPUs are not the
> binding resource on this project and multi-GPU parallel training is available; the
> GPU-hour figure above is a *consequence* of this decision, not its reason. The reason:
> with C3 and C4 switched on at the same time, the headline claim — "C1+C2 change the
> class ranking without disturbing `Z_gen`" — is no longer separable; the gain spreads
> across four sources and a reviewer cannot ask which one worked. Phase 1 runs first so
> that the contribution can be attributed cleanly.
>
> **Phase 2 follows it, and does not wait long.** Both scenarios are live. If Phase 1
> lands as expected, C3/C4 run directly on top of it, warm-started from the Phase 1
> checkpoint. If Phase 1 misses its bar, Phase 2 becomes a **rescue arm** — C3 in
> particular is the only arm that can undo the damage the hard mask does in pediatrics,
> where regions are small and often fail to reach the feature grid at all (§07, the
> empty-region cliff). It has a price: because ρ requires a mask at inference, the
> mask-free-inference property is lost, and that trade is reported separately. The one
> hard condition is that Phase 1's *measurement* closes before Phase 2 is mixed in.

`01 · starting point`

## Where we stand

The ground the new architecture is built on. One evaluator, identical volumes.
The C47 rows appear in none of the existing reports — they finished on 25 August.

| run | single change | peds 27 | peds routed | CT-RATE 27 | adult-18 | adult loss |
|---|---|---|---|---|---|---|
| V1 | baseline (old labels) | 0.8005 | 0.7651 | 0.7954 | 0.8151 | −0.0423 |
| V1-seed1 | seed only · **on a different label set**, §14 | 0.7933* | — | — | — | — |
| V2 | re-extracted labels | 0.7974 | 0.7657 | 0.8000 | 0.8187 | −0.0387 |
| V3 | + empty-region verdicts | 0.7972 | — | 0.8003 | 0.8189 | −0.0385 |
| V4 | + top-K focal pooling | 0.7971 | — | 0.8005 | 0.8187 | −0.0387 |
| V5 | + multi-scale grid 24³ | **0.8019** | 0.7662 | 0.7925 | 0.8086 | −0.0488 |
| C16 | joint, 14.7k adult, *fabricated labels* | 0.7881 | 0.7596 | 0.7991 | 0.8378 | −0.0196 |
| C47mix | joint, 42.5k adult, real labels | 0.7964 | 0.7608 | 0.8228 | 0.8400 | −0.0174 |
| C47harm | + label harmonisation | 0.7886 | 0.7587 | 0.8271 | **0.8442** | **−0.0132** |
| ARC-CT | published adult model — **the line to beat** | — | — | — | 0.8574 | — |

*Scored peds val **n=1,772** (the split says 1,774; 2 volumes have no label row) ·
CT-RATE val n=3,002. ***** V1-seed1 was measured on the old label set (it matches
0.7961, not 0.8005) — that is why the band is 0.0028; do not read 0.0072 off the table.
**How the seed band may be used was rewritten from scratch in §14: it cannot serve as a
decision threshold.***

> **bar status**
>
> **The pediatric bar depends on the scale, and on the scale we declared it has not
> been cleared yet.** The expectation in the notes is *"the pediatric AUC ceiling is not
> that important… but hopefully +80"*. On the 27-class scale V5 is **0.8019**; but in
> §11 we declared **measurable 23** as the headline metric, and there V5 = **0.7957**.
> So the bar looks cleared on the 27-class scale and not cleared on the honest one.
> The genuinely hard job is still the **adult bar**: 0.8442 today, gap 0.0132 ± 0.0022 (§13).

`02 · directives`

## Directive → decision mapping

The **principal directives** from the notes and the decisions that correspond to them.
**I diverged explicitly in two places** (marked `[DIVERGED]`); in three more I accepted
the directive and corrected its *implementation* (the MLP form of r_c, the global query
being an addition rather than a split, cropping to the body rather than the lungs).
The table does not cover every directive; the closing bullets are handled in the body
(§01, §09, §13, §14).

> **from the notes · the framing sentences**
>
> > *"The indication does not tell the model what is present. It tells the model where and what to interrogate more carefully, while an indication-independent representation preserves comprehensive interpretation of the CT."*
> > *"Anatomy determines where a finding can occur, while indication determines which findings and visual patterns deserve increased attention."*
>
> These two sentences are the constraint on the whole design. Every mechanism will be
> tested against them: if the indication *suppresses* a finding, the mechanism is wrong.

| directive | decision | where |
|---|---|---|
| Use the indication in the QFormer pathology query embeddings | The main contribution. **Cross-attention** (query → indication token sequence) **+ FiLM** channel gating. A token sequence, not a single pooled vector. | §05 |
| Keep the unconditional CT pathway; `Z = Z_gen + g⊙Z_ind`, learned gate | **Dual pathology bank**: 27 unconditioned + 27 conditioned, weights tied. Fusion by **concat** (his preference), gated sum as an ablation. Phase-1 bank: 67 slots / 40 distinct queries. | §04, §05 |
| Relevance gate `w=1+βr` — **never** multiply by `r` | Adopted exactly. A low-relevance class keeps weight 1, it does not go to 0. Incidental-finding protection is built into the architecture. | §06 |
| Do not condition the anatomy queries | Adopted exactly — the conceptual separation stays clean. (It was already so in rev.1.) | §04 |
| Split the two global queries: general / clinical-question | Accepted, but as an **addition, not a split**: both inherited globals stay in `Z_gen` and the clinical-question query is added as the **40th distinct query** — otherwise `Z_gen` averages over 38 tokens and is no longer identical to ARC-CT (§04). | §04 |
| Add 4 free indication queries (for what falls outside the 27 classes) | **Moved to Phase 2 in rev.7.** Our cohort is full of exactly those cases (osteosarcoma surveillance, post-transplant, thymoma) — but C4 is the largest shortcut surface in the design and its "contributes nothing when the indication is absent" claim is not mechanised (§08). | §08 |
| Age-band embedding + sex | Ordinal-cumulative band embedding; age/sex join the indication token sequence as **context tokens** — not a separate branch. | §09 |
| Indication dropout 20–30%, mismatched-indication test | Dropout **30%** + a counterfactual consistency loss — **restored in rev.5**, but applied to the **prediction logits**, not to `Z_gen` (§12). | §12, §14 |
| Keep the existing losses, add `L_ind = Σ(1+βr)·BCE` | Accepted — that BCE already exists (`loss_pertoken`) — but it is **not "one line"**: with the dual bank the slice becomes P=54 and the broadcast raises; which bank it supervises must also be stated (§12). | §12 |
| Do not hand-code `r_ic`, learn it from the embeddings | Learned exactly as asked — but using **the MLP option of the two given in the notes**: the dot-product form is untrainable with frozen towers (§06). | §06 |
| Concatenation beat cross-attention (on a modest dataset) | The ablation ladder was made mandatory. Our indications are 10× longer, but the peds **training** set is only 7,042 volumes (the training portion of the 8,817 cohort) — so the warning applies to us **exactly**. | §14 |
| Do not condition every layer; keep it in the last blocks | That is already our situation: the ResNet is never touched, conditioning lives only in the Q-Former queries — i.e. the 4-block bridge. | §04 |
| Use free text, not categories | Adopted exactly. Structured concept extraction (symptom / context / target condition / anatomy) also remains as a Phase-2 option. | §05 |
| Weak localisation loss (`L_localization`) | The artefact is ready (no segmentation needed) but it is **not free**, and rev.4 wired it wrongly: the cache maps *report findings* to regions, not the indication → it is applied to the unconditioned bank only (§12). | §12 |
| **Go mask-free for peds / drop the anatomy queries** | **DIVERGED.** Instead of deciding by hand I leave it to a **learned gate**: the mask is always supplied and its hardness λ is learned. If λ→0, that is a *finding*. The manually-disabled version remains as an ablation arm. Reason: §03 Fact 2. | §07 |
| **We could also drop pneumothorax** | **DIVERGED.** CT is the reference standard for pneumothorax; the low score is not caused by the modality but by the region-8 shell we derived ourselves. Keep it, and make it C3's declared test case. | §11 |
| Drop the adult-only classes from pediatric scoring; arterial wall calc? | All four (Emphysema 3 · Coronary 5 · Hiatal hernia 18 · **Arterial wall calc 34**) are loss-masked on the peds side and absent from the peds score → headline **measurable 23**. They leave the score, not the model. The definition fix's 0.654→0.735 gain is reported separately as a *label-quality* finding. | §11 |
| Apply **cropping** for pediatrics instead of downsampling directly — infants have very small lungs | Accepted. §10 was rewritten accordingly: crop to the body box → resample to 192³. Two technical additions: it must crop to the **body, not the lungs** (otherwise regions 9 and 10 disappear), and the voxel size must be handed back to `H_dem` as a token (otherwise absolute scale is lost). | §10 |
| Resolution??? | **Measured, and the prediction was refuted.** The dominant effect is not padding but cropping (data is discarded in 72% of volumes; median padding 5.2%). The peripheral-class prediction was indistinguishable from zero (contrast +0.0154, CI [−0.076, +0.087]). | §10 |

> **dropped since rev.1**
>
> The **M3 — age-conditioned prevalence prior** (logit adjustment) of the earlier version
> does not appear in Dr. Kurugol's design, and his `r_ic` relevance mechanism already does
> the defensible part of that job. It was demoted to **optional and lowest priority**;
> it is not in Phase 1.

`03 · reading the code`

## Three structural facts I found in the code

These are not measurements, they are source-code readings. None of the three appears in
the existing reports, and all three directly determine the architecture.

> **Fact 1 · `train_stage2.py:663–674`**
>
> **Every pathology query is already a per-class classifier head — it is trained, but it
> is never read at test time.**
>
> ```
> path_tokens = qf_tokens[:, A : A+P]        # 27 pathology query tokens
> path_lat    = normalize(path_tokens)
> sims_pos    = (path_lat * pos_embs).sum(-1)   # each query against its own "X" prompt
> sims_neg    = (path_lat * neg_embs).sum(-1)   # each query against its own "No X" prompt
> loss_ptok   = CE(softmax([pos,neg]/τ), label) # weight 0.5, ON in both configs
> ```
>
> So `query[10+p]` is literally the detector for class `p`, and it lives in the text
> latent space. But `evaluate.py` never uses `qf_tokens`: it uses only (a) the mean over
> the 39 queries, and (b) the mask-pooled organ latent. **We are throwing away a trained,
> calibrated, class-specific readout head.**
>
> Design consequence: the pathology queries are the *right* place to inject the
> indication — they are already separated per class and already in the shared
> image–text space. And reopening that readout head costs **zero extra training**.

> **Fact 2 · `evaluate.py:294–312`**
>
> **The "routing penalty" does not measure the Q-Former's hard mask. It measures a
> completely different readout head — one that was never trained against class prompts.**
>
> `_routed_probs` replaces the global latent with `organ_lat`. But in training,
> `organ_lat` is aligned *only* through `soft_clip_infonce(organ_lat, region_lat)`, i.e.
> to *region sentences*. At evaluation it is then scored against *class prompts*
> ("Pneumothorax." / "No Pneumothorax.") — a target bank it has never seen.
>
> This is a train/test objective mismatch, and it predicts the observed pattern exactly:
> the largest penalty falls where the region text is sparsest and most heterogeneous —
> Pneumothorax (−0.3369, region 8) and Bone lesion (−0.2222, region 9).
>
> **Consequence: the claim "hard masking hurts" is not established.** All three
> hypotheses are alive and all three are separable:
>
> - **H1 · mask quality** — the pleural shell we derived is too thin. *Test:* thicken region 8, re-measure the penalty.
> - **H2 · checkpoint selection bias** — **rev.6 correction:** the docstring of `train_stage2.py` says "mask-free", but `run_validation` *does* pass the masks into the Q-Former; selection is therefore already oracle-masked. The only thing excluded from selection is the `_routed_probs` organ-pool readout. This weakens H2 and strengthens Fact 2; the test is still run over intermediate checkpoints.
> - **H3 · readout objective mismatch** — the above. *Test:* add a class-prompt loss to `organ_lat`, or stop reporting this metric as a "routing metric".
>
> The proposal document's "we are building on sand" warning is therefore premature. All
> three measurements take less than a week and everything needed for two of them is
> already on disk.

> **Fact 3 · `anatomy_qformer.py:93`**
>
> **Soft routing already exists — by accident.**
>
> ```
> any_key = out.any(dim=-1, keepdim=True)
> out = torch.where(any_key, out, torch.ones_like(out))   # empty region → fully free
> ```
>
> If a region never reaches the feature grid, the query silently becomes an unrestricted
> global query. Measured miss rates on the current 10-region masks: right middle lobe
> **5.6%**, central airway **1.5%** (rev.4's 10%/3% were figures from the abandoned
> 13-region scheme). Even in adults, on CT-RATE validation, airway **4.1%** and heart
> **4.5%** — i.e. the heart is slightly worse than the airway.
>
> So the system already behaves as "if the mask is unreliable, fall back to global" — it
> just does so *unconditionally* and *unlearned*. Our contribution is not inventing a new
> mechanism: it is **turning an unprincipled fallback into a learnable gate**. That is a
> much easier claim to defend.

> **[FIGURE 1]** Three readout heads, one of them never read. Three paths leave the same
> tower. R1 is the primary metric and it is what protects the adult gate. R2 is trained
> but never used at evaluation — a free-gain candidate and the place where C1's effect
> will show. R3's training target (region sentences) differs from its test target (class
> prompts); the "routing penalty" is measured on precisely that dashed arrow.

> **how these three facts connect to the directives**
>
> - **Fact 1 → `L_ind` is one line.** The indication loss proposed in the notes,
>   `L_ind = Σ(1+βr_ic)·BCE(ŷ_ic, y_ic)`, is offered on the grounds that "you aren't
>   really introducing a completely new training objective, since you already supervise
>   each pathology query against positive and negative class prompts". That is exactly
>   right — that BCE exists in our code as `loss_pertoken` and is enabled. `L_ind` is a
>   reweighting of it.
> - **Fact 1 → the conditioned readout head is ready.** The "conditioned head" of the
>   notes is our R2; it is already trained, `evaluate.py` simply does not read it.
> - **Fact 2 → hold the "mask-free peds" decision.** The main empirical justification for
>   abandoning masks — the −0.03 routing penalty — is not measuring the hard mask.
>   Discarding ARC-CT's published core contribution on the strength of a confounded
>   measurement would be an expensive mistake. Three hours of measurement separates them
>   (§17, Phase 0).
> - **Fact 3 → the gate is already half-built.** The "empty region → unrestricted query"
>   behaviour is the accidental, unconditional form of mask-free peds. Making it learnable
>   turns "masked or mask-free?" from a design decision into a *measurement*.

`04 · the system`

## Architecture: the whole picture

The inherited ARC-CT path does not change. The context path is new. The critical
structural property: the pathology representation is produced in **two copies** — one
that never sees the indication, one that does — and the two are merged at the end.
That is what preserves an unexpected finding.

### Query bank: Phase 1 → 40 distinct queries, 67 slots · Phase 2 → 44 / 71

| group | count | conditioned | attention restriction | role |
|---|---|---|---|---|
| Anatomy | 10 | no | masked to a single organ | organ representation — untouched |
| Pathology · general | 27 | no | masked to the union of organs | unconditioned path · safety net |
| Pathology · conditioned | 27 | **yes** | same mask, same weights | indication path · main contribution |
| Indication query **(C4 · PHASE 2)** | 4 | **yes** | unrestricted | catches what falls outside the 27 classes — **absent** in Phase 1 |
| Global · general (inherited, 2 rows) | 2 | no | unrestricted | holistic CT summary |
| Global · clinical question **(NEW)** | 1 | **yes** | unrestricted | the 40th distinct query — it does not replace the inherited globals, it is added to them |

*Phase 1: **67 attention slots · 40 distinct query vectors · 1 new query** (the
clinical-question global). With C4 it becomes 71 / 44 / 5. The conditioned pathology bank
**shares weights** with the general one (the same `q_c`, one passing through conditioning
and one not), so the 27 extra slots introduce no new parameters. Cost (Phase 1):
**1.72× relative to ARC-CT** (67/39); the marginal share of the dual bank is 67/40 = 1.68×;
self-attention among queries is O(Q²) but negligible at this scale. **But the two banks do
not stay separate on their own** — see the next part.*

### Isolating the unconditioned pathway — four leakage paths

The dual bank guarantees nothing by itself. Inside the Q-Former, conditioned and
unconditioned queries share one forward pass, and information can leak to the
unconditioned side through **four distinct paths** — all four silently, with no error
raised. This is the same class of failure as the defect catalogue: the measurement runs,
a number comes out, and the number is wrong.

> **leak 1 · self-attention**
>
> `QFormerBlock` runs self-attention across all queries in every block
> (`self.self_attn(h, h, h)`, no mask). So the unconditioned pathology queries *see* the
> conditioned ones and pick up indication information indirectly. One block later the
> "unconditioned" bank is no longer unconditioned.
>
> **Fix:** a group mask on self-attention. The rule is one-directional — **the
> unconditioned group may not attend to any conditioned query**; the conditioned group
> may attend to everything.
>
> ```
> unconditioned group (39) = anatomy(10) + pathology-general(27) + global-general(2)
> conditioned   group (28) = pathology-conditioned(27) + global-clinical(1)      # +4 indication queries in Phase 2
> self_attn_mask[i, j] = MASKED   if i ∈ unconditioned  and  j ∈ conditioned
> ```

> **leak 2 · the λ gate — the dangerous one**
>
> C3's gate had been written as `λ_q = softplus(a_q + w_q·[H̄_C ; ρ])`. But λ determines
> *where in the image a query may look*. If the λ of the unconditioned queries depends on
> the indication, then the **attention geometry of the unconditioned path is being steered
> by the clinical question** — a deeper leak than self-attention, because it directly
> changes which voxels are seen.
>
> **Fix:** λ is split in two.
>
> ```
> λ_q^gen = softplus( a_q + w_q·[ ρ ; H̄_dem ] )          # unconditioned bank: does NOT see the indication
> λ_q^ind = softplus( a_q + w_q·[ ρ ; H̄_dem ; H̄_ind ] )  # conditioned bank: clinical question included
> ```

> **leak 3 · fusion initialisation**
>
> If `Z_final = W[Z_gen ; Z_ind]` is initialised randomly, then at step 0 the model is not
> ARC-CT — half of `Z_ind` mixes in as noise and the "we start from the published weights"
> claim collapses.
>
> **Fix:** `W` starts as **[ I ; 0 ]**. The final piece of the zero-init discipline:
> conditioning gates at 0, β = 0, fusion blind to the `Z_ind` half ⇒ **at step 0,
> Z_final = Z_gen = ARC-CT, bit for bit.**

> **leak 4 · pooling — added in rev.5, and the most dangerous**
>
> Rev.4's three-path list was **incomplete**. At the end of `QFormer.forward`:
>
> ```
> if self.pool == "mean":
>     pooled = q.mean(dim=1)     # mean over the query axis = CROSS-TOKEN MIXING
> # RAC_QFORMER_POOL=mean, enabled in both configs
> ```
>
> And that `pooled` is exactly **R1 / Z_gen** — the thing labelled "PRIMARY METRIC — the
> adult gate" in this document's own Figure 1 — **and it is also the checkpoint-selection
> metric**. In a 67-slot bank, `q.mean(dim=1)` averages the conditioned tokens in as well,
> so:
>
> - `Z_gen` silently becomes indication-conditioned — even with the other three fixes fully applied;
> - the checkpoint-selection metric becomes indication-conditioned;
> - §15's headline experiment falsifies its own prediction, and the natural (wrong) reading is "one of the three fixes is buggy".
>
> **Fix:** pooling becomes group-restricted — `Z_gen = mean(q[:, gen_idx])`, and `Z_ind`
> is the relevance-weighted normalised pool over the conditioned indices.
> `RAC_QFORMER_POOL=mean` is removed from both env files. This does **not** follow from
> "FFN and LayerNorm are per-token"; it is a separate item.

> **the bit-identity claim — corrected form**
>
> Rev.4 said "at step 0 the model is bit-identical to ARC-CT". **It was not**, for four
> independent reasons: (i) the published peds model averages over *all* 39 queries, while
> `Z_gen` would average over 38 (the second global query moved into the conditioned group),
> shifting every token's weight from 1/39 to 1/38; (ii) the self-attention group mask
> changes the softmax denominator of the unconditioned rows (39 keys → 38); (iii) `−8`
> instead of `−∞` (the leak table in §07); (iv) the global grad-clip of 0.5 — the new
> modules have zero-*output* but non-zero *gradients*, which inflates the total norm so
> the original parameters take a smaller step.
>
> **Fix:** both inherited global queries stay in `Z_gen` and the clinical-question query is
> added as the **40th distinct query** (so `Z_gen` is again averaged over 39 tokens, the
> same as ARC-CT); and the claim becomes an *assertion*: run the 39-query and 67-slot
> models on the same batch and verify `‖Z_final − img_lat_ARC‖ < 1e-5` before any training
> starts. If it does not pass, the claim comes out of the paper.

> **the paths that can be proven clean**
>
> The remaining paths are structurally safe and it matters to say so: **image
> cross-attention** (both groups look at the same unconditioned `F`), **FFN**,
> **LayerNorm** (including `norm_out` — all of them over the last axis, per token),
> **residual connections** (index-aligned addition) and **dropout** (0.0, and it does not
> mix tokens anyway). There is no batch-wise statistic anywhere inside the Q-Former. So
> after **four** fixes, `Z_gen`'s independence from the indication is structural *for the
> forward pass*.
>
> **But not for the parameters.** The conditioned and unconditioned banks share every
> block weight, so gradients from the conditioned branch also update the operators the
> unconditioned branch uses. At fixed weights `Z_gen(I) = Z_gen(I')` holds (which is why
> §15 is safe), but saying "the unconditioned path *is* ARC-CT" is not correct: its
> parameters are shaped by the indication from step 1 onward. That is something to
> **state**, not to hide.

> **terminology**
>
> "Unconditioned" here means **independent of the indication**, not *independent of
> context*. Age and sex deliberately enter both paths: they are properties of the patient,
> not the clinical question. Age entering `λ^gen` is C3's entire rationale (age = spatial
> scale) and it does not affect the counterfactual experiment in §15 at all — in that
> experiment the patient is fixed and only the indication varies.

> **[FIGURE 2]** Indication-Conditioned Anatomy-Routed Attention. The image path (grey)
> does not change: the ResNet is not conditioned, conditioning lives only in the Q-Former
> bridge. The context path (amber) carries the indication as a *token sequence*; the age
> and sex embeddings are not a separate branch but additional tokens of the same sequence
> — which is how age can also influence spatial attention. On the right, two pools:
> `Z_gen` has never seen the indication, `Z_ind` has; they are joined by concatenation.
> The box at the bottom lists the four independent layers that prevent a finding from
> disappearing because of the indication.

`05 · the main contribution`

## C1 — Indication → pathology query

### C1 — Cross-attention + FiLM, dual-bank  `[proposal]`

**INTUITION**

The Q-Former's pathology queries are 27 separate questions of the form "is X present in
this volume?". Tell a radiologist *"cystic fibrosis follow-up"* and they look with a
different set of questions: mucus plugging, bronchiectasis, tree-in-bud move to the
front. That is exactly what we are doing — the indication text reshapes the *vector* of
each pathology query.

The critical point: this is not giving the classifier a hint. The query vector is
*attention's question*; changing it changes *what the model extracts* from the image,
not how the extracted thing is scored.

**FORMAL DEFINITION**

```
H_ind = CXR-BERT(indication)                # token SEQUENCE — not pooled
H_dem = [ e_age(b) ; e_sex ; e_int ]        # demographic tokens
H_C   = [ H_ind ; H_dem ]                   # what the conditioned queries see

# 1 · the spatial question: the query looks at the indication tokens
q̃_c = q_c + g_x · CrossAttn( q_c , H_C , H_C )       # g_x zero-init scalar gate

# 2 · the channel question: which kind of feature matters
q̃_c = γ(H̄_C) ⊙ q̃_c + β(H̄_C)     γ = 1 + ε·tanh(W_γ·), ε = 0.2, W zero-init
# THE ε BOUND IS REQUIRED: γ = 1+tanh(·) ∈ (0,2), so at tanh→−1, γ→0 and the query
# was erased entirely — the "residual, q is never erased" safety layer was a lie.

# 3 · query the image — the anatomical restriction REMAINS, but C3 softens it (§07)
z_c = CrossAttn( q̃_c , F , F ;  role mask + λ gate )

# unconditioned copy: same q_c, no conditioning, λ^gen gate
z_c^gen = CrossAttn( q_c , F , F ;  same mask + λ^gen )

# 4 · fusion — W starts as [ I ; 0 ]
Z_final = W [ Z_gen ; Z_ind ]                        # at step 0 = Z_gen = ARC-CT
```

**WHAT CHANGED SINCE REV.1 AND WHY**

The earlier version had FiLM only, and my reasoning was "the indication collapses into a
single vector". **That is wrong for our corpus.** With a median of 21 words, the
difference between *"History of metastatic osteosarcoma, evaluate pulmonary nodules"* and
*"Incidental 5 mm nodule follow up"* is lost in a pooled vector. Cross-attention onto the
token sequence preserves it.

But there is counter-evidence and it must be taken seriously: in the study cited in the
notes, on 3D chest CT, simple concatenation beat cross-attention, and the authors
attributed that to insufficient data. Our cohort is 8,816 volumes, but the cross-attention
parameters are fit only on the training portion: 7,042 volumes (1,774 validation). So we
are squarely in the "modest data" regime. The 42,544 adult volumes in joint training do
not compensate — CT-RATE has an indication in only 49.8% of cases with a median of 2
words, so the effective corpus that can teach a rich indication conditioning is in
practice those 7,042 volumes. That is why the ladder in §14 is mandatory, not optional.

**WHY THE PATHOLOGY QUERY AND NOT SOMEWHERE ELSE**

- **Not the ResNet** — the general CT representation must stay independent of the indication; the notes also say "do not condition every layer". In our design the ResNet is never touched.
- **Not the anatomy queries** — anatomy is independent of the clinical question. Letting the indication reshape the anatomy queries opens the door to the model inventing anatomy from text instead of from the image.
- **The pathology queries, yes** — §03 Fact 1: they are already separated per class and already live in the text latent space. Modulation here is both interpretable (‖Δq_c‖ = "how much did this indication open this class") and directly connected to the existing loss function.
- **And there is an observed rationale.** In the §13 breakdown, the three classes carrying the *resolvable* part of the adult loss — Mosaic attenuation, Pulmonary fibrotic sequela, Lung nodule — are the three classes whose meaning is *age-dependent*. All three started bit-identical to ARC-CT weights and still eroded: joint training degrades adult skill on classes whose meaning diverges between cohorts. C1's age-conditioned modulation is the natural address for that. **Associational, not causal** — the same three classes are also on the annotator-threshold-mismatch list (§13).

`06 · the safety mechanism`

## C2 — Relevance gate

### C2 — w = 1 + β·r · non-suppressing weighting  `[proposal]`

**THE SHARPEST DETAIL IN THE DIRECTIVES**

The notes are explicit: do **not** multiply the query by `r_c`, because that suppresses
incidental findings. Use `w_c = 1 + β·r_c` instead. Low-relevance classes keep weight 1,
they do not drop to 0.

```
r_c = σ( MLP[ E(I) ; E(P_c) ; E(I)⊙E(P_c) ] )   # learned relevance, NO hand-coded table
w_c = 1 + β·r_c ,   β = softplus(b), b init ≈ −6   # β ≥ 0 STRUCTURALLY
Z_ind = Σ_c w_c·z_c / Σ_c w_c                   # NORMALISED weighted pool
        ⊕ z_ind-query ⊕ z_global-clinical
```

Three design details, all three necessary:

- **MLP, not a dot product.** The form `σ(E(I)ᵀE(P_c)/τ)` is elegant but **untrainable**: both embeddings come from a frozen tower, with no learned parameter in between. Relevance would be pinned to CXR-BERT's pretrained similarity structure. A small MLP (768·3 → 128 → 1) opens it up; the `E(I)⊙E(P_c)` term supplies a cheap interaction signal.
- **Normalised pool.** With a raw `Σ w_c z_c`, the pool's norm would depend on *how many classes are relevant* — a broad indication ("chest pain") would systematically produce a larger vector than a narrow one. Dividing by `Σ w` preserves the relative emphasis and removes the magnitude artefact.
- **β is learned but sign-constrained — a rev.5 fix.** In rev.4 β was a free scalar, and together with `L_ind = Σ(1+βr)·L_ptok` it had a **runaway solution**: because `L_ptok ≥ 0`, `∂L/∂β = Σ r·L_ptok > 0` *always*, so gradient descent drove β negative from the first step. Result: `w < 1` — the headline safety property violated within the first hundred steps of optimisation, followed by the `Σw → 0` pole. With `β = softplus(b)`, `w ≥ 1` becomes a **structural guarantee** rather than a hope.
- **And the loss is normalised too**, for the same reason: `L_ind = Σ_c (1+βr_c)L_ptok,c / Σ_c (1+βr_c)`. Un-normalised, shrinking β would just be scaling down a positive loss; normalised, `∂L/∂β ∝ Cov_c(r_c, L_c)`, so β genuinely learns to promote classes that are *relevant and poorly fit*.

`E(P_c)` does exist in our code as `pos_embs` — but **not every step**: it is computed
under `@torch.no_grad` and refreshed only at each validation (400–1200 steps). So it
requires no extra forward pass, but the relevance MLP's class basis is **detached and
stale**; the MLP absorbs that drift, and the refresh interval must be reported.

**EXPECTED BEHAVIOUR**

| indication: "lung cancer follow-up" | r_c | w_c (β=1) |
|---|---|---|
| Lung nodule | 0.95 | 1.95 |
| Lymphadenopathy | 0.88 | 1.88 |
| Pleural effusion | 0.61 | 1.61 |
| Atelectasis | 0.48 | 1.48 |
| Cardiomegaly | 0.08 | 1.08 |
| Coronary calcification | 0.03 | 1.03 |

*The example from the notes. The bottom two rows are the whole point of the mechanism:
irrelevant classes are **still measured**. The indication says "pay particular attention
to these", it does not say "ignore everything else".*

**HOW IT GETS KILLED**

The mismatched-indication test. Give a CT that contains a nodule the indication
*"rule out pneumonia"*. The nodule prediction must not disappear. If it does, β is too
large or the conditioning is too strong — this single test confirms or refutes C2's
reason for existing.

`07 · where I diverged`

## C3 — Routing gate `[PHASE 2]` and the cliff fix `[PHASE 1]`

**Split in two in rev.7.** Closing the empty-region cliff is a *bug fix* and belongs in
Phase 1; the learned λ gate is a separate *claim* and belongs in Phase 2. Tying them
together was unnecessary — the cliff must be fixed whether or not C3 ever exists.

> **from the notes**
>
> *"Maybe just go mask-free with pediatrics (TS masks and the resolution is a big problem)"*
> *"For Ped. maybe just omit the anatomy queries, but use pathology queries (with indication)"*

The evidence supporting this suggestion is real and strong: in pediatrics the
mask-routed AUC is ~0.03 below the mask-free one in every model, regions fail to reach
the feature grid in small children, we derived the pleural shell ourselves and it carries
−0.34 on pneumothorax, and TotalSegmentator costs 80–133 seconds per volume.

Even so I **do not recommend turning the masks off by hand**, for four reasons:

- **The main evidence is confounded.** §03 Fact 2: that −0.03 measures not the hard mask but an organ-pool readout head that was never trained against class prompts. Discarding ARC-CT's published core contribution on the strength of a confounded measurement would be an expensive mistake.
- **But first, a distinction: removing the mask is not removing the Q-Former.** Rev.4 conflated these. The Q-Former stays either way. There are two independent decisions: (a) should the hard mask in cross-attention go, and (b) do the 10 anatomy queries earn their slots. (a) is a mechanism question, (b) a capacity question — a mask-free model *with* anatomy queries is perfectly coherent.
- **If the anatomy queries go, ARC-CT's published contribution closes on the pediatric half.** This is a more concrete cost than the "one model, two populations" claim: the per-organ alignment loss (`L_org`) depends on the anatomy queries and the organ masks. Without them there is nothing for `L_org` to attach to on the pediatric side — the pediatric model is then no longer ARC-CT, it is a plain Q-Former CLIP.
- **The cost is already sunk.** 8,816 pediatric masks were produced and are on disk. Not using them saves nothing today. The real saving is in the CT-RATE *training* masks (~1,740 GPU-hours), and that decision depends on the outcome of this measurement.

> **PHASE 1 · bug fix — independent of C3**
>
> Before returning, `build_role_mask` turns an empty region **fully free** via
> `where(any_key, out, ones)`. So an anatomy query silently becomes a global query
> whenever its region does not reach the grid — with no warning, and at the measured
> rates: right middle lobe 5.6%, central airway 1.5%, and even in adults airway 4.1%.
>
> **This must be fixed without waiting for the learned gate.** Two parts: (i) ρ must be
> computed *before* the override and returned separately (read afterwards it yields
> ρ = 1.0 for an empty region, the exact inverse of the signal); (ii) the empty-region
> rate must be logged in training and evaluation, so that "was anatomy routing actually
> in force" is an answerable question. In this form it is not an architecture change but
> a **visibility fix**.

### C3 — Learned hardness: leave the decision to the model  `[Phase 2 — optional arm]`

**FORMAL DEFINITION**

```
# TODAY: the mask is a wall
logits[q,n] = −∞  if n ∉ R(q)
# and if the region is empty the mask lifts ENTIRELY — a hard cliff

# PROPOSAL: the wall becomes a slope whose steepness is learned
ρ_r  = |R(r) ∩ grid| / 12³                  # how much of this volume region r occupies
λ_q^gen = softplus( a_q + w_q·[ ρ ; H̄_dem ] )            # unconditioned bank: does NOT see the indication
λ_q^ind = softplus( a_q + w_q·[ ρ ; H̄_dem ; H̄_ind ] )    # conditioned bank: clinical question included
# a_q init = 8 → both start at ≈ a hard mask
logits[q,n] −= λ_q · (1 − m[q,n])
# λ → ∞   ARC-CT's hard mask (exact recovery)
# λ → 0   mask-free Q-Former = the notes' "mask-free peds"
```

> **rev.5 · two fatal implementation errors fixed**
>
> **1 · The empty-region guard neutralises the gate.** Before returning,
> `build_role_mask` does `out = where(any_key, out, ones)`; in an empty region `m ≡ 1`,
> hence `λ·(1−m) = 0` — *identically, for every λ*. So C3 does nothing in the very cliff
> case it exists for, and `∂L/∂λ = 0`. **Fix:** remove the guard. This is safe, because a
> uniform `−λ` applied to all keys cancels in the softmax — a fully penalised query
> converges *continuously* to the unrestricted global query. (The guard stays for
> `has_mask=False` rows; there it is the correct behaviour.)
>
> **2 · Read after the guard, ρ gives the inverted signal.** Rev.4 said "ρ is already
> computed (`out.sum(-1)`)"; no such line exists, only `out.any(-1)`. And computed
> *after* the `where`, an empty region yields **ρ = 1.0** — maximum coverage, the exact
> opposite of the truth. **ρ must be computed BEFORE the override and returned separately.**

Because `H̄_dem` contains the age band, **λ can vary with age**: the model can learn to
loosen the mask in a two-year-old and keep it tight in a forty-year-old. But `λ^gen` does
not see the indication (§04).

> **a_q = 8 is not "hard" — the leak table**
>
> With N = 12³ = 1728 keys, the attention mass escaping the region for `k` allowed tokens:
>
> ```
> k = 40 (adult lung)        →   1.40%
> k = 10                     →   5.45%
> k =  2 (trachea, age 2)    →  22.4%
> k =  1                     →  36.7%
> # on the 24³ grid (N=13824), k=2 → 69.9%
> # to keep the leak under 1% at k=2 you need λ ≥ 11.4
> ```
>
> So starting at `a_q = 8` is **not** starting from the published behaviour. **Fix:**
> `a_q` is initialised coverage-aware (`a_q = log((N−k_q)/(0.01·k_q))`, from the empirical
> ρ), and the leak table is reported.

**WHY NOT IN PHASE 1 — two real risks**

**1 · Its motivation comes from a confounded measurement.** C3's rationale is that the
mask-routed AUC is ~0.03 lower in pediatrics; but §03 Fact 2 shows that penalty measures
a different readout head, not the hard mask. Putting C3 on the delivery list before
Phase 0/1–3 separates them would be building a mechanism on a rotten rationale.

**2 · It requires a mask at inference — and that breaks a published property.** λ depends
on ρ; ρ comes from the mask. But one of ARC-CT's selling points is **mask-free inference**.
Put C3 in as written and the model now demands a mask at inference — not a performance
regression but a *claim* regression, which is worse. There are three exits and one must be
chosen: (a) `λ^gen` is fed only by age/sex and ρ is used only in training; (b) ρ is
predicted from the image; (c) λ is frozen to a per-query constant after training. None of
these happens by itself.

**WHY IT IS STILL VALUABLE — IN PHASE 2**

"We turned the masks off in pediatrics and it got better" is an engineering note.
**"The model learned by itself to abandon anatomy routing in children under five, and λ
increases monotonically with age"** is a finding — and it is direct evidence for why age
conditioning is needed. Same experiment, same cost, a far stronger claim.

**PRE-REGISTERED PREDICTION**

(a) λ increases monotonically with age band — loosest in the youngest band.
(b) The gain concentrates in regions 8 and 9 — Pneumothorax, Pleural thickening, Bone
lesion. If it appears elsewhere, the mechanism is not working for the reason we claim,
and it is reported that way.

**AGE-BASED SWITCHING — THE DISCRETE VERSION**

An alternative: look at the age in the metadata and, in a child, turn the anatomy queries
and the masking off entirely, working with global + pathology queries only. This is C3's
**discrete special case**: λ ∈ {0, ∞}, switched at an age threshold.

In favour: simple, interpretable, no learned gate that can go wrong. Against, three
things: (i) the threshold is arbitrary — measured region survival degrades *continuously*
with body size, there is no jump at 18 (or at 5); (ii) the pooled query set would vary
per sample, which collides with the isolation guarantee in §04 (`Z_gen`'s composition must
be fixed); (iii) the `L_org` cost above.

**Recommendation:** C3's continuous gate stays as the mechanism, and the discrete age
switch becomes an arm in the ladder. If the discrete version wins, we report it — and then
instead of "the model learned to turn anatomy routing off by age" we say "we turned it off
by age and it worked". The second sentence is weaker but it is not *wrong*; and we are not
targeting high pediatric scores anyway.

> **[FIGURE 3]** Removing the cliff. The right-hand column is the real problem: in the
> current code, when a region never reaches the grid the mask lifts *entirely*, so the
> query stops being an anatomy query and becomes a global one in a single step — with no
> warning. In children this happens 5.6% of the time for the right middle lobe. The
> learned gate makes the same behaviour continuous and inspectable; because `a_q` is
> initialised large, training starts from the published behaviour.

`08 · beyond the 27 classes`

## C4 — Free indication queries `[PHASE 2]`

### C4 — 4 learned clinical-question tokens  `[Phase 2 — optional arm]`

**REV.7 · WHY IT WAS MOVED BACK FROM PHASE 1 TO PHASE 2**

**C4 is the largest shortcut surface in the design.** Four free queries: no anatomical
mask, conditioned only on the indication, feeding the prediction directly. If the model
wants to learn "the indication says X → predict X", this is the cheapest place to do it —
the masked pathology queries at least have to look somewhere anatomically plausible,
these do not. It will not lower AUC; it will probably *raise* it. That is precisely the
problem. It is the first thing that will break in the mismatched-indication test.

Also, §08's own claim that "it contributes nothing when the indication is absent" is not
mechanised: `H_C` still carries the age/sex tokens even when the indication is dropped, so
`Q̃_I ≠ Q_I` and the queries keep producing something derived from age. That is the second
reason for Phase 2.

**WHY IT IS STILL VALUABLE**

In the notes this appears as an "eventually", a foundation-model extension. In our cohort
it is needed *now*, because a children's hospital's indications are full of concepts that
fall outside the 27 classes:

- *"History of thymoma, evaluate mediastinal recurrence"*
- *"Persistent cough following stem cell transplantation"*
- *"Osteosarcoma surveillance"* — the most frequent short indication in our cohort
- *"Evaluate postoperative complication"* — Post-surgical change is at 33% prevalence

None of these is a class name; but all of them say *where to look*. Twenty-seven fixed
queries cannot carry that information.

**FORMAL DEFINITION**

```
Q_I = { q_1^I , … , q_4^I }                # learned, not tied to a class
Q̃_I = Q_I + CrossAttn( Q_I , H_C , H_C )   # conditioned from free text
Z_I = CrossAttn( Q̃_I , F , F )             # unrestricted — NO anatomical mask
                                           # because which organ it belongs to is unknown
```

These enter the `Z_ind` pool, not `Z_gen`.

**INTERPRETABILITY BONUS**

The attention maps of the four queries show directly "where the model translated the
clinical question to" — and because they are not constrained by a class label, the most
striking visuals of the §15 experiment will probably come from here.

`09 · the direct answer to the question`

## Exactly where age and sex enter the model

The question in the notes: *"Age band embeddings falan modele tam nasıl girecek?"*
The answer in three sentences: **as extra tokens in the context sequence**, with an
**ordinal-cumulative parameterisation**, and with the **sex interaction starting from
zero**.

> **the one-sentence answer**
>
> Age and sex are **not a separate branch**: they are appended as two or three extra
> tokens to the indication token sequence (`H_C = [H_ind ; H_dem]`, with each demographic
> token projected to 768-d). This way C1's cross-attention, C2's relevance score and C3's
> λ gate — **all three** — see the age. Had it been a separate MLP branch, age would only
> have entered the classifier prior; whereas age's real job is *spatial scale*, which is
> C3's job.

### The two corpora barely overlap in age

BCH pediatric — **8,870 accessions** · CT-RATE train (n=47,137)

| age | pediatric n | % | CT-RATE n | % |
|---|---|---|---|---|
| 0–1 | 301 | 3.4 | 0 | 0.0 |
| 1–2 | 222 | 2.5 | 7 | 0.0 |
| 2–5 | 807 | 9.1 | 0 | 0.0 |
| 5–10 | 1291 | 14.6 | 0 | 0.0 |
| 10–13 | 1097 | 12.4 | 0 | 0.0 |
| 13–18 | 2701 | 30.5 | 0 | 0.0 |
| 18–21 | 1305 | 14.7 | 747 | 1.6 |
| 21–25 | 740 | 8.3 | 2102 | 4.5 |
| 25–30 | 250 | 2.8 | 3660 | 7.8 |
| 30–40 | 114 | 1.3 | 10049 | 21.3 |
| 40–50 | 26 | 0.3 | 9095 | 19.3 |
| 50–60 | 13 | 0.1 | 7490 | 15.9 |
| 60–70 | 3 | 0.0 | 6880 | 14.6 |
| 70+ | 0 | 0.0 | 7107 | 15.1 |

*Ages come from metadata: `search_CHESTCT_8k.xlsx` column M on the pediatric side, DICOM
`PatientAge` for CT-RATE. The highlighted rows mark the single bridge. **Count
clarification:** the age table is over **8,870 accessions**; the modelling cohort is
**8,816 volumes** (8,817 retrieved), the split is 7,042 / 1,774, and the scored set is
**1,772**. In this document "cohort" = 8,816.*

### Why not a scalar — three reasons

- **Prevalence is neither linear nor monotone in age.** Coronary calcification is effectively zero until 30 and then explodes. Mucus plugging peaks in children with CF and falls in young adults. A scalar age forces the network to produce these shapes from a single linear direction.
- **Age is a proxy for spatial scale.** Body diameter varies by 2.4× across the cohort while the grid is fixed at 192×192×96. The same structure occupies a very different number of tokens in a six-month-old and a sixteen-year-old. The routing table depends on exactly that token count — which is why age must enter *routing* via C3, not only the classifier prior.
- **There is a gap between the corpora.** CT-RATE has 7 volumes under 18 out of 47,137. A scalar age forces the model to extrapolate linearly across a range where it has no data. Bands represent the gap explicitly.

### Ordinal-cumulative parameterisation

A plain embedding table forgets that the bands are *ordered*: the distance from 2–5 to
5–10 is learned the same way as the distance from 2–5 to 60+. To fix this we write the
embedding as a cumulative sum of increments:

```
e_age(b) = e_0 + Σ_{j ≤ b} softplus(Δ_j)      # Δ ∈ R^{9×64}, small init
# ordering is built into the structure (softplus ≥ 0 ⇒ a monotone walk)
# the spacings are LEARNED — non-linear, but not arbitrary either

e_sex ∈ R^{16}                              # 3 rows: F / M / unknown
e_int[b, s] ∈ R^{32}   ZERO-INIT, wd ×10     # starts purely additive

# rev.5 FIX: these cannot be appended directly to H_ind (a 768-d CXR-BERT token
# sequence) — 64/16/32 ≠ 768. Each passes through its own projection:
P_age: R^64 → R^768 ,  P_sex: R^16 → R^768 ,  P_int: R^32 → R^768  (P_int zero-init)
H_dem = [ P_age·e_age ; P_sex·e_sex ; P_int·e_int ]     # 3 tokens, each 768-d
```

| band | range | clinical name | peds n | adult n | note |
|---|---|---|---|---|---|
| 0 | < 1 y | infant | 301 | 0 | largest scale difference |
| 1 | 1–2 y | toddler | 222 | 7 | CT-RATE's 7 are probably metadata errors |
| 2 | 2–5 y | preschool | 807 | 0 | trachea/oesophagus drop off the grid |
| 3 | 5–10 y | school age | 1291 | 0 | |
| 4 | 10–13 y | pre-adolescent | 1097 | 0 | |
| 5 | 13–18 y | adolescent | 2701 | 0 | peak of the pediatric cohort |
| 6 | 18–25 y | young adult | 2045 | 2849 | **the single bridge band** — where transfer happens |
| 7 | 25–40 y | adult | 364 | 13709 | |
| 8 | 40–60 y | middle age | 39 | 16585 | atherosclerosis begins here |
| 9 | 60+ y | elderly | 3 | 13987 | effectively empty on the pediatric side |

*Ten bands, covering both corpora. **The model never sees a cohort/site flag** — only the
age. This is deliberate: a cohort flag is the easiest shortcut and it would immediately
refute the "one model, two populations" claim.*

> **design consequence**
>
> This turns "27.6% of our pediatric cohort is adult" from a defect into a *feature*. The
> model does not see two labels, it sees a *continuum*. The 69-year-old with congenital
> pulmonary stenosis and the six-month-old both sit on the same age axis, and the model
> reads each from its own band.

`10 · the question mark in the notes`

## Resolution and body size

It appears twice in the notes with question marks. This section was **rewritten from
scratch in rev.5**: the earlier version used numbers copied from training-time log files,
and one class had the wrong sign.

> **retracted claim**
>
> Rev.4 said: *"Lung nodule 0.6727 → 0.6647, −0.0080 (the training loop's own evaluator) —
> the textbook example of a small focal finding, and it got worse with finer resolution."*
> That table came from `runs/peds_finetune_v*/best_auc_update*.txt`, i.e. **the training
> loop's own evaluator**. On the single-evaluator `eval_matrix` that §01 uses, the same
> difference is **+0.0021**. The sign is inverted, and the narrative built on it is void.
>
> Also, rev.4 read Lung nodule's −0.0080 as "degradation"; that class's **measured seed
> band is 0.0118** — so both numbers were inside their own noise.

### What we measured: the 24³ grid's gain comes from four classes

| class | V3 (12³) | V5 (24³) | Δ | val pos. | note |
|---|---|---|---|---|---|
| Bone lesion or fracture | 0.5700 | 0.6153 | +0.0453 | 293 | best-supported gain — but CI 0.115, below the reportability floor (§11) |
| Hiatal hernia | 0.7540 | 0.7882 | +0.0342 | 18 | CI width 0.223 |
| Arterial wall calcification | 0.7354 | 0.7659 | +0.0305 | 34 | CI width 0.185 |
| Coronary artery wall calc. | 0.7796 | 0.8062 | +0.0266 | 5 | CI width **0.394** |
| Pulmonary metastases | 0.6907 | 0.7126 | +0.0219 | 94 | |
| Lung nodule | 0.6726 | 0.6747 | +0.0021 | 910 | seed band 0.0118 → uninterpretable |
| 15 classes | — | — | negative | — | worst: Pulmonary cyst −0.0168 |
| **MEAN (27)** | 0.7972 | 0.8019 | **+0.0047** | — | |
| **MEAN (measurable 23)** | 0.7941 | 0.7957 | **+0.0016** | — | **inside the seed band** |

*Single evaluator (`eval_matrix/peds`), the same 1,772 volumes. The four dropped classes
(Emphysema, Coronary, Hiatal hernia, Arterial wall calc) contribute **+0.0886** in total
from V3→V5, i.e. **+0.0033** of the 27-class mean — **71%** of the 27-class gain.*

> **the correct conclusion, and stronger than rev.4's**
>
> **The multi-scale grid produces no measurable gain on adequately supported classes.**
> 71% of the +0.0047 in the 27-class mean comes from four classes with 3, 5, 18 and 34
> positives in validation respectively — one of which (Coronary) has a confidence interval
> 0.394 wide. On the measurable 23 the gain is **+0.0016**, inside the seed band.
>
> There is no need to say "Lung nodule got worse" for this — that claim was both wrong and
> unnecessary. And this also explains what makes V5 "the sharpest trade": its pediatric
> gain is unmeasurable while its adult loss (−0.0488) is real.

### The direction of the fix: cropping — this is a directive, not a hypothesis

> *"We can apply cropping for pediatrics data, especially infants etc they have really
> small lungs — instead of downsampling directly, we can apply cropping and get the most
> useful data in that CT."* — Dr. Kurugol

This is the directive form of what I had written as a hypothesis in rev.5, and it also
corrects the mechanism in one place. **The problem is not "downsampling", it is the fixed
box.** The pipeline already resamples to 1.5×1.5×3.0 mm — infant CTs are often acquired
finer than that, so what happens there may technically not even be downsampling. The real
waste is the fixed 192×192×96 box: an infant's body occupies a small part of it and the
rest is padding. And because the 12³ feature grid is computed over the *whole box*, most
of the 1728 tokens look at padding.

```
# today: each token ≈ 24 × 24 × 24 mm physically
192×192×96 input  →  12³ grid  →  1 token = 16×16×8 voxels

# if an infant's body is ~90×90×60 voxels, the anatomy fits into ~5.6×5.6×7.5 tokens;
# the lungs are 2–3 tokens wide. Making the grid finer does not change this,
# because it makes the padding finer in the same proportion.

# after cropping: the same 12³ grid, but all of it looks at tissue
crop to body box (90×90×60)  →  resample to 192×192×96  →  1 token ≈ 11 mm
# token density (tokens per organ) becomes roughly constant across ages
```

> **rev.5 · MEASURED — and the dominant effect turned out to be CROPPING, not padding**
>
> Rev.4/5 said *"padding median 18%, up to 77% in the smallest"*; that number was a
> comment line in `dataset.py`. I have now measured it properly on a random sample of 500
> volumes — and **both figures pointed the wrong way**:
>
> ```
> # PEDS_NPZ: full FOV, 1.5×1.5×3.0 mm, VARIABLE size (the 192³ box is applied at load time)
> in-plane (H)  median 215   min  95   max 333      # box 192
> slices   (D)  median  92   min  32   max 698      # box  96
>
> relative to the 192×192×96 box:
>   in-plane   71.2% CROPPED (data discarded)    27.8% padded
>   slices     41.4% cropped                     57.8% padded
>
> fraction of the box filled with real data   median 94.8%   quartiles 72.1% / 100%   min 8.4%
> ⇒ PADDING fraction                          median  5.2%   max 91.6%
> ⇒ volumes with data DISCARDED               72.2%
> ```
>
> So the median padding is **5.2%, not 18%**, and in **72% of volumes** the problem is not
> padding but the fixed box cutting the periphery away. Median in-plane is 215 voxels
> against a box of 192 — about 12 voxels, i.e. **~35 mm**, discarded per side going
> outward from the centre.

> **this does not weaken the cropping directive — it doubles it**
>
> The fixed box does damage in **two different ways**, and body-normalised cropping fixes
> both at once:
>
> - **In larger children (72%):** the periphery is cut off. What gets cut is exactly **region 9 (chest wall, ribs, spine)** and **region 10 (upper abdomen)** — the two outermost regions of our schema. Crop to the body box and what is cut becomes air and table, not tissue.
> - **In infants (the extreme 8%):** padding rises to 91.6%, with nine tenths of the grid looking at empty space — the case Dr. Kurugol pointed at, and it is real, just *rare*.
>
> And this may explain our weakest class. The lowest pediatric AUC is
> `Bone lesion or fracture` = **0.5700**, and its only address is region 9 — the region
> being cut away. Right behind it, `Pleural thickening` 0.6784, region 8. Both peripheral.
>
> **Pre-registered prediction:** per-class AUC should be inversely related to the volume's
> *cropping* fraction, and the effect should concentrate in the region 8/9/10 classes.

> **prediction run · REFUTED**
>
> 1,772 pediatric validation volumes, the V2 checkpoint, cropping fraction computed per
> volume and the bottom and top tertiles compared:
>
> ```
> cropping fraction   median 0.217   quartiles 0.000 / 0.428   max 0.968
> low-crop n=594 (≤0.031)   ·   high-crop n=590 (≥0.368)
>
> PERIPHERAL (5 classes)  mean AUC difference  = +0.0070
> CENTRAL    (13 classes) mean AUC difference  = −0.0084
> CONTRAST                                     = +0.0154
>    patient-level bootstrap 95% CI  [−0.0757, +0.0873]   P(≤0) = 0.407
> ```
>
> **Indistinguishable from zero.** Moreover the spread *within* the central classes
> (Pulmonary metastases **+0.1243** … Pulmonary fibrotic sequela **−0.0774**) is an order
> of magnitude larger than the contrast itself — so what was measured is noise.
>
> And the headline prediction came out **in the wrong direction**:
> `Bone lesion or fracture`, the only class in region 9, scores **0.5013** in the low-crop
> group and **0.5613** in the high-crop group. That is, the model is at chance level
> precisely where its region is *fully present*. This class's weakness is not a cropping
> artefact.

> **the limits of the test — what died and what did not**
>
> This test compares **subgroups**, not the intervention: high-crop volumes are
> systematically larger/older children, so their disease mix differs too. It is a cheap
> screen, not a firm rejection. The real test is to apply the cropping and re-evaluate.
>
> **What survives:** the measurement itself — in 72% of volumes the fixed box discards
> data, which is a fact. Cropping removes that waste, and it is Dr. Kurugol's directive.
> **What falls:** the expectation that it "will fix the peripheral classes". So cropping is
> no longer a *high-expected-gain* item; it is **cheap, directive-backed, and of uncertain
> return**. It stays in Phase 2 and the 2×2 ablation arm will measure it.

```
1. find the body bounding box   # HU threshold + largest connected component
2. CT and MASK are cropped to the SAME box  # both through the same code, the same box
3. resample to 192×192×96       # token density becomes constant across ages
4. the voxel size is added to the context   # an extra token in H_dem: mm/voxel
```

- **Crop to the body, not the lungs.** The sentence says "small lungs", but if the box is cropped to the lungs then **two regions of our schema disappear**: region 9 (chest wall, ribs, spine) and region 10 (upper abdomen) — both outside the lungs. And region 9 is the sole address of Bone lesion (13.9% prevalence). The crop is to the *body* box; the arms are excluded.
- **Absolute scale is lost — but it can be given back.** After cropping, voxel size varies per patient, so "a 5 mm nodule" is no longer a fixed number of voxels. That is a real risk for size-criterion findings. **The solution is already in the architecture:** the mm/voxel value enters `H_dem` as an extra token. Age was already a *proxy* for spatial scale; now we hand over the scale itself.
- **The mask and the CT must travel the same path.** This defect has already happened once: when the mask and the CT reached 192³ by different geometric routes, in a 186-voxel-wide child the mean HU under the lung label came out at −281 while the background sat at −528. The crop box must be computed **once** and applied to both through the same function.
- **Partly a substitute for C3.** If cropping raises the rate at which regions reach the grid, C3's λ gate has less to loosen in children. The two partly solve the same problem, so the ablation must be **2×2**: cropping on/off × gate on/off. Otherwise one's gain is credited to the other.

**Cost — corrected in rev.5.** The earlier version said "the npz must be regenerated";
that was wrong and implied deriving from raw data. The real situation is far cheaper:
**the raw NIfTI (3.5 TB) is untouched** and **TotalSegmentator does not re-run** (that was
the expensive item; `TS_MASKS_PEDS` is on disk with 8,816 entries). Cropping is a
transformation of the npz we already have:

```
CT   : existing npz (full FOV) → body box → 192³        # pure CPU transformation
MASK : TS_MASKS_PEDS → THE SAME box → 192³              # no GPU, just a resample

# two implementation options:
(A) write new npz            # +125 GB disk, fast loading
(B) box = 6 integers/volume  # ~0 disk, crop+resize at load time
    stored as a sidecar      # switch to (A) if the dataloader CPU becomes the bottleneck
```

(B) is probably the right start: 8,816 × 6 integers is a trivial file, and the existing
`_pad_crop_hwd` already runs at load time — so there is no architecture change.

`11 · class decisions`

## Class schema: the answer to the question

> **from the notes · a direct question**
>
> *"Adult only olan classları (emphysema, hiatal hernia ve calcification) pediatric
> scorelamadan çıkart. Coronary wall cal. kesin çıkacak ama arterial wall calc?
> Pneumothorax'ı da çıkarabiliriz."*
> ("Drop the adult-only classes — emphysema, hiatal hernia and calcification — from
> pediatric scoring. Coronary wall calc. is definitely out, but arterial wall calc?
> We could drop pneumothorax too.")

First, the critical distinction: "drop from scoring" and "drop from the model" are not the
same thing, and the difference here determines the adult gate. Those classes are our
**strongest** on the adult side: Arterial wall calc. 0.9365, Coronary 0.9321, Hiatal
hernia 0.8363. Delete the head and that skill is deleted.

> **first, let us prevent a number confusion — two cohorts, two separate numbers**
>
> There is **no** situation that could be read as "it was 0.9 in the ARC-CT paper and now
> it is in the 0.7s". The 0.9 is an *adult* number and the 0.7 a *pediatric* one — not the
> same number having fallen, but two different things measured in different cohorts. The
> adult side is holding:
>
> | class | ADULT · CT-RATE val | | | PEDIATRIC · BCH val | | |
> |---|---|---|---|---|---|---|
> | | ARC-CT | C47mix | C47harm | V1 | V2 | C47harm |
> | Arterial wall calc. | 0.9365 | 0.9367 | 0.9313 | 0.7419 | 0.7339 | 0.7036 |
> | Coronary artery wall calc. | 0.9321 | 0.9280 | 0.9305 | 0.8214 | 0.7806 | 0.8641 |
> | Emphysema | 0.8261 | 0.8044 | 0.8142 | 0.9930 | 0.9911 | 0.9966 |
> | Hiatal hernia | 0.8363 | 0.8107 | 0.8220 | 0.7749 | 0.7496 | 0.8107 |
>
> *The adult columns have moved at most **−0.0143** relative to ARC-CT and all four are
> inside their own confidence intervals. So joint training has not yet damaged adult skill
> on these classes — pediatric prevalence is 0.3–1.6%, so the gradient is tiny to begin
> with. The Emphysema value of 0.99 in the pediatric columns comes from **3 positives**.*
>
> **But the mechanism behind the concern is real and visible in the table:** Arterial wall
> calc's *pediatric* number falls run over run — 0.7419 → 0.7339 → **0.7036** — while the
> adult data grows from 14.7k to 42.5k. This is not degradation, it is **evidence**: the
> model is progressively learning the class's *adult* meaning (atherosclerotic
> calcification) and the pediatric "positives" (ligamentum arteriosum, catheter tracts,
> cartilage) do not fit it. The two labels really are different diseases. The loss mask
> cuts this prospectively and stops the pediatric score from being a meaningless number.

> **the mechanism — and this is exactly what you meant**
>
> All 27 heads stay. A loss mask is applied **per (volume, class)**: on pediatric volumes
> the relevant columns produce no gradient, on adult volumes they train at full strength.
> That machinery is already written — the NaN-cell masking fix (defect #6) exists for
> exactly this. **The pediatric side does not touch the model at all on these classes** —
> neither gradient nor score — while on the adult side the head keeps training at full
> strength. In reporting, the pediatric headline is given over **"measurable 23"**
> (27 minus Emphysema, Coronary, Hiatal hernia, Arterial wall calc). For V2 the
> comparison is: 27-class 0.7974 · 23-class 0.7944 — a difference of about one seed band.

| class | peds prev. | peds val pos. | adult prev. | adult AUC | decision |
|---|---|---|---|---|---|
| Emphysema | 0.3% | 3 | 19.4% | 0.8261 | peds loss-masked · absent from peds score |
| Coronary artery wall calc. | 0.3% | 5 | 25.5% | 0.9321 | peds loss-masked · absent from peds score |
| Hiatal hernia | 0.9% | 18 | 14.3% | 0.8363 | peds loss-masked · absent from peds score |
| Arterial wall calc. | 1.6% | 34 | 28.4% | 0.9365 | **OUT** — loss-masked in peds, absent from peds score. Rationale below; but the expected adult gain is small (measured: −0.0052). |
| Pneumothorax | 2.3% | 37 | 0.5% | — | **STAYS.** C3's declared test case. |

### Arterial wall calcification: why we are dropping it — and why this will not save the adult gate

In adults this class means atherosclerotic calcification of the systemic arteries and it
is one of ARC-CT's strongest: **0.9365**. In children it is, in practice, **a different
disease**. The sentences our own evidence file cites:

*"Normal variant calcification of the ligamentum arteriosum"* ·
*"prominent tracheal and bronchial cartilage calcifications"* ·
*"calcification in the left brachiocephalic vein"* ·
*"calcifications … reflecting the tract of the Port-A-Cath"* ·
*"Peripheral calcified densities … in the left pulmonary arteries"*

Venous calcification, catheter tracts, airway cartilage, normal variants, and pulmonary
rather than systemic arteries. After the definition was narrowed, prevalence fell from
2.78% to 1.61% — but we cannot claim the remaining 34 positives are atherosclerotic;
children do not have atherosclerosis. **Producing gradient on a different disease under
the same name blurs a 0.9365-quality adult detector.**

> **but set the expectation correctly**
>
> This will not save the adult gate. We measured it: this class's loss is **−0.0052** in
> C47harm and **+0.0002** in C47mix — i.e. it is already almost fully preserved. The reason
> is simple: pediatric prevalence is 1.6%, so the pediatric gradient is tiny already. The
> loss mask is the right decision, but it is right for reasons of **scientific hygiene**
> (a 34-positive number belonging to a different disease should not be in a headline mean),
> not for the adult gate.
>
> The same holds for the other three rare classes: Coronary −0.0016, Emphysema −0.0119,
> Hiatal hernia −0.0143 — all four inside their own confidence intervals. Even masked, the
> total adult gain is at most **~0.0018**, probably zero. The gap is −0.0132.
> **The adult gap is somewhere else** — §13.

> **rev.5 · reportability floor — a separate and independent rule**
>
> Class selection is decided by **disease identity**; the positive count is reported but is
> not the criterion. Were it the criterion it would be inconsistent: Pericardial effusion
> stays with 84 positives (CI width **0.200**) while Arterial wall calc leaves with 34
> (0.185).
>
> Instead, a second **pre-declared** rule: no mechanism claim is built on any class whose
> 95% bootstrap CI is wider than **0.10**. Measured (n=1,772): Coronary 0.394 · Hiatal
> hernia 0.223 · Pericardial effusion 0.200 · Arterial wall calc 0.185 · Pulmonary
> metastases 0.164 · **Pneumothorax 0.161** · Pulmonary cyst 0.144 · Mass 0.118 ·
> **Bone lesion 0.115**.
>
> **The cost is explicit: Pneumothorax and Bone lesion are below the floor** — so C3's
> region-8/9 prediction *cannot be tested* on this cohort at this n. We say this now, not
> after seeing the result. At n₁ ≤ 20 the percentile bootstrap degenerates (Emphysema, with
> 3 positives, returns a spurious 0.007 width); there Hanley–McNeil intervals are reported.

### Pneumothorax: why I recommend not removing it from the model

**CT is the reference standard for pneumothorax** — it is the modality that catches occult
pneumothoraces invisible on a radiograph. The argument "it cannot be determined properly
from CT" would cause trouble in review.

The two real problems we measured are both **ours**: prevalence 2.3% (199 positives, 37 in
validation) and the routed penalty of **−0.3369** — because it is routed to the pleural
shell we derived ourselves. The evidence is in the neighbouring class: `Pleural effusion`
goes to `[8,2,5]`, i.e. pleura *plus* backup lobes, and its penalty is small.

If a radiologist's objection concerns *report* reliability (a small apical pneumothorax may
not be mentioned in the report), that is a label-noise argument — and the fix is the same
masking mechanism, not deleting the class.

`12 · the objective`

## Losses

The principle from the notes: keep the three existing losses, add the indication loss on
top. In our case this is an even smaller change than expected.

```
L = L_clip                      # Jaccard-smoothed soft InfoNCE (FN_WEIGHT 0.3) — UNCHANGED
  + L_cls                       # prompt BCE, over Z_final — UNCHANGED
  + L_org                       # organ latent ↔ region sentence — UNCHANGED
  + L_ptok^gen                  # UNCONDITIONED bank — as in ARC-CT, unweighted
                                #   (this is what protects the adult gate and the step-0 story)
  + Σ_c (1+β·r_ic)L^ind_ptok,c / Σ_c(1+βr_ic)   # CONDITIONED bank — relevance-weighted, normalised
  + λ_loc · L_loc               # weak localisation — to the UNCONDITIONED bank (below)
  + λ_cf · Σ_c 1[r_c<τ]·‖ŷ_c(I) − ŷ_c(I')‖²   # counterfactual — over the PREDICTIONS
```

### L_ind is one line

The notes say *"you aren't really introducing a completely new training objective"* and
that is right — `loss_pertoken` already exists. But it is **not "one line"** (rev.4's
claim): the current code slices `qf_tokens[:, A:A+P]` assuming P=27; with the dual bank
P becomes 54 and the broadcast against `pos_embs` raises. Which bank it supervises must
also be stated. **Rev.5 decision:** the unconditioned bank keeps ARC-CT's unweighted loss
(the adult gate and the step-0 identity depend on it), and the relevance-weighted term is
applied as a *separate* loss to the conditioned bank. This also makes the C2 ablation clean.

### Counterfactual consistency — restored, over the right quantity

The notes say *"counterfactual indication training occasionally"*: the same CT, different
indications; *the general findings should remain stable while attention changes*. In the
first draft I enforced this with a loss term (`‖Z_gen(I) − Z_gen(I')‖²`).

In rev.3 I removed that term, on the grounds that after isolation `Z_gen(I) − Z_gen(I')` is
identically zero, so the penalty optimises something that is identically zero. **The
reasoning is right but about the wrong quantity.**

The safety claim is not about `Z_gen`, it is about the **prediction**: "the pneumothorax on
a CT ordered for pneumonia must not disappear". The prediction comes from
`Z_final = W[Z_gen ; Z_ind]`, and `W`'s right block is trained **unconstrained** after
step 0, while `Z_ind` is fully indication-dependent. So isolation proves `∂Z_gen/∂I = 0`
but says nothing about `Z_final`.

**Rev.5:** the term returns, over the logits of the low-relevance classes. In addition, the
ratio `‖W_ind‖ / ‖W_gen‖` is monitored and reported throughout training as a diagnostic —
it shows how far the conditioned path has taken over the fusion.

### L_loc: the artefact is ready, but it is not free

The artefact is ready: `region_cache.json` holds LLM-assigned sentences for 10 regions per
volume (55,966 keys). **But rev.4 wired it wrongly.** The cache maps *report findings* to
regions, not the indication. Pulling the conditioned query there **directly contradicts**
the headline experiment of §15: the report of the same CT is identical under all three
indications, so the target is fixed — whereas C1's whole purpose is for attention to
*change*. Our own loss would have flattened the headline figure.
**Rev.5:** `L_loc` is applied to the **unconditioned bank only** — there, "look where the
findings are" is a coherent, indication-independent target.
And it is not "free": `return_attn=True` disables the fused SDPA path and materialises
~77 MB of attention tensor per batch.

### What we do not train

We do not train the model to *reproduce* the indication — that is the warning in the notes.
The indication is a condition, not a target. Otherwise the text channel learns to explain
itself instead of the image.

### Parameter groups

| component | state | LR | rationale |
|---|---|---|---|
| 3D ResNet-18 | training | 1.1e-5 | same as ARC-CT · not conditioned |
| CXR-BERT body | FROZEN | — | the zero-shot prompt space depends on it |
| Text LoRA (report path) | training | 1e-6 | should shift the prompt space only slightly |
| Indication path | **SEPARATE FROZEN COPY** | — | rev.5: "use the same tower without LoRA" is **not implementable** — LoRA mutates the module in place and `to_text_latent` is also trained. A genuinely frozen path means a second CXR-BERT copy with its own projection (memory + a second forward pass). |
| Anatomy queries (10) | training | 1.1e-5 | the region meanings changed in peds |
| Adult pathology rows (18) | training | 1.1e-6 | **0.1×** — protect adult skill |
| New pathology rows (9) | training | 1.1e-5 | learned from scratch |
| C1–C4 + context encoder | training | 1.1e-5 | zero-init, let them learn at full speed |

`13 · the project's hard constraint`

## Closing the adult gate

Target: beat ARC-CT (0.8574) on the 18 CT-RATE classes, or bring the difference into the
noise band. Our best today is 0.8442 — a gap of 0.0132.

> **a change of framing**
>
> Until now we have posed this gap as a "recover the skill we lost" problem. **Context
> conditioning turns it into a winning problem.** At least three of the 18 adult classes
> (coronary calcification, arterial calcification, emphysema) are directly age-driven, and
> age is **100%** available in CT-RATE. The context path carrying age therefore brings
> *new signal* on the adult side too — both to the query conditioning (C1) and to the
> routing gate (C3). The realistic route to beating 0.8574 is this, not loss recovery. The
> indication, by contrast, is present in only 49.8% of CT-RATE, so most of the adult gain
> must come from age.

### Exactly where the gap is — per class for the first time

We distributed the 0.0132 over the 18 classes. The table below gives the raw deltas; which
of them can be separated from noise is determined by the **paired** test (end of this
section).

| class | ARC-CT | C47harm | Δ | peds prev. | note |
|---|---|---|---|---|---|
| Mosaic attenuation pattern | 0.8316 | 0.7513 | −0.0803 | 15.3% | the three largest losses |
| Pulmonary fibrotic sequela | 0.7114 | 0.6623 | −0.0491 | 19.7% | the three largest losses |
| Lung nodule | 0.7593 | 0.7199 | −0.0394 | 52.8% | the three largest losses |
| Peribronchial thickening | 0.8328 | 0.8143 | −0.0185 | 15.1% | |
| Bronchiectasis | 0.8097 | 0.7946 | −0.0151 | 13.0% | |
| Hiatal hernia | 0.8363 | 0.8220 | −0.0143 | 0.9% | ~absent in peds |
| Lung opacity | 0.8761 | 0.8641 | −0.0120 | 28.7% | |
| Emphysema | 0.8261 | 0.8142 | −0.0119 | 0.3% | ~absent in peds |
| Lymphadenopathy | 0.7871 | 0.7782 | −0.0089 | 7.9% | |
| Atelectasis | 0.8060 | 0.7990 | −0.0070 | 32.6% | |
| Cardiomegaly | 0.9445 | 0.9385 | −0.0060 | 2.7% | |
| Medical material | 0.9236 | 0.9183 | −0.0053 | 39.8% | |
| Arterial wall calcification | 0.9365 | 0.9313 | −0.0052 | 1.6% | ~absent in peds |
| Consolidation | 0.9276 | 0.9250 | −0.0026 | 6.0% | |
| Coronary artery wall calc. | 0.9321 | 0.9305 | −0.0016 | 0.3% | ~absent in peds |
| Pleural effusion | 0.9737 | 0.9726 | −0.0011 | 4.3% | |
| Pericardial effusion | 0.8869 | 0.9029 | +0.0160 | 4.6% | gained |
| Interlobular septal thickening | 0.8318 | 0.8564 | +0.0246 | 5.7% | gained |

*CT-RATE validation, n=3,002. The ARC-CT column is the published `augmd_seed0` run
(0.8574). Net: (−0.2783 + 0.0406)/18 = −0.0132.*

> **RETRACTED · rev.4's headline finding**
>
> Rev.4 claimed: *"schema.py deliberately re-initialised 5 rows in the 30→39 query
> surgery… these 5 rows carry 77% of the adult loss."* **This claim is wrong and is
> retracted.** I measured it.
>
> I compared the query tensors of `arcct_seed0.peds27.pt` and `arcct_seed0.pt` row by row:
>
> ```
> 18 adult pathology rows  →  maxAbsDiff = 0.000e+00  (bit-identical)
> 2 global rows            →  maxAbsDiff = 0.000e+00  (bit-identical)
> anatomy rows 0–6         →  maxAbsDiff = 0.000e+00  (bit-identical)
> anatomy rows 7, 8, 9     →  re-initialised  # pleura(derived) · chest wall · upper abdomen
> 9 new pathology rows     →  fresh, std 0.01992  # the source tensor's std is 0.019892
> ```
>
> **No pathology row was ever re-initialised.** The surgery that was actually performed
> gave fresh rows to the three anatomical regions that are *new* in the schema; everything
> whose meaning survived was copied. I had taken the five-name list from a plan file that
> was never executed; the surgery code is not in `schema.py` (there is not even a tensor
> there) but in `arcct/arcct-bch/model/qformer_surgery.py`, and `--reinit` is a
> command-line argument.
>
> Three things that depended on it also fell and were removed from the document in rev.5:
> the lever "revert the re-init" (there is nothing to revert), rev.4's Phase-0 item
> (~15 GPU-hours; removed from the table, and the slot number now belongs to a different
> measurement), and §11's "it should have been on that list" rhetoric.

> **the replacement finding — narrower but solid**
>
> **All 18 adult rows started bit-identical to the ARC-CT weights and 0.0132 was still
> lost.** So the adult gap is not an *initialisation* event but a *training* event: the
> warm start did not protect them, the pediatric gradient eroded them.
>
> Which deltas can be separated from noise? **Rev.5 measured this wrongly**: I compared each
> delta against the half-width of a single arm's own confidence interval, whereas the two
> models are measured on *the same 3,002 volumes*. The right tool is a **paired**
> bootstrap — the sampling noise largely cancels and the interval narrows. Paired,
> patient-level (1,304 patients, 1,000 replicates):

| class | Δ (C47harm − ARC-CT) | paired 95% CI | excludes zero |
|---|---|---|---|
| Mosaic attenuation pattern | −0.0803 | [−0.1031, −0.0591] | yes |
| Pulmonary fibrotic sequela | −0.0491 | [−0.0657, −0.0331] | yes |
| Lung nodule | −0.0394 | [−0.0535, −0.0260] | yes |
| Peribronchial thickening | −0.0186 | [−0.0298, −0.0083] | yes |
| Bronchiectasis | −0.0151 | [−0.0271, −0.0030] | yes |
| Hiatal hernia | −0.0143 | [−0.0294, −0.0000] | borderline |
| Lung opacity | −0.0120 | [−0.0176, −0.0062] | yes |
| Emphysema | −0.0119 | [−0.0231, −0.0013] | yes |
| Lymphadenopathy | −0.0090 | [−0.0169, −0.0005] | yes |
| Arterial wall calcification | −0.0052 | [−0.0089, −0.0014] | yes |
| Atelectasis · Cardiomegaly · Medical material · Consolidation · Coronary · Pleural effusion | −0.0070 … −0.0011 | contains zero | no |
| Pericardial effusion | +0.0159 | [+0.0036, +0.0293] | yes (gain) |
| Interlobular septal thickening | +0.0246 | [+0.0077, +0.0402] | yes (gain) |

*Reference: `results/adult_gate/augmd_seed0` (our own e3 reproduction, macro-18 =
**0.857398**). There is a second reference on disk — `arcct/reference`, the original
Bilkent run, **0.858346**; FINAL_REPORT records the two separately (delta −0.0009). We use
our own reproduction for the gate, because it was measured with the same code and
environment, so the difference reflects the model.*

**12 of 18 deltas exclude zero, 10 of them losses.** The resolvable losses carry
**−0.2549** of the gross negative total (−0.2785), i.e. **91.5%**. So computing a share is
*legitimate* — rev.5's "no percentage can be reported" ruling was over-cautious and came
from the wrong tool.

And the three classes whose meaning is age-dependent (Mosaic, Fibrotic sequela, Lung
nodule) carry **−0.1688** on their own: **61%** of the gross loss and **71%** of the net —
and all three are individually significant.

> **what remains for C1 — association, not cause**
>
> The three classes carrying measurable loss are the three whose meaning is age-dependent.
> Mosaic: small-airways disease / chronic PE in adults, post-infectious bronchiolitis
> obliterans in children. Fibrotic sequela: IPF in adults, post-radiotherapy/chemotherapy
> scarring in children. Lung nodule: an incidental screening nodule in adults,
> *osteosarcoma metastasis surveillance* in children.
>
> This is still a good motivation for C1's age-conditioned modulation — but no longer as
> "we are fixing a re-init mistake"; in much simpler terms: *joint training erodes adult
> skill on classes whose meaning diverges between cohorts, and age is the observable axis
> of that divergence.*
>
> **Associational, not causal.** The three classes are also on the annotator-threshold
> mismatch list. Confounders below.

> **unseparated confounders**
>
> - **Label harmonisation is of the same magnitude.** The largest per-class movement between the two joint runs is Interlobular septal thickening: 0.7805 → 0.8564 (**+0.0759**) — and that is not an "age-dependent meaning" class, it is a threshold-mismatch class. It also moves Mosaic by 0.0287 on its own, i.e. **32%** of the −0.0803.
> - **Training length.** C47harm peaked at 4,800 (stopped at 8,800), C47mix at 16,800 (stopped at 20,800) — a factor of 3.5. The paired bootstrap of the macro-18 difference between the two runs is **+0.0041, 95% CI [0.0013, 0.0070]**: labels + length alone produce a significant difference.
> - **Ceiling effect.** Classes with a lower ARC-CT baseline lose more. These three average 0.767 in ARC-CT against 0.876 for the other 15.
> - **Shared-backbone drift** — not class-specific.
>
> **The discriminating experiment:** a 2×2 of {harmonised / published labels} × {fixed step
> budget}, with ≥2 seeds per cell — 8 runs, ~120 GPU-hours. Early stopping **off**,
> otherwise length confounds it again.

| # | lever | what it does | cost | expected effect |
|---|---|---|---|---|
| 1 | Zero-init everywhere | C1–C4 are no-ops at step 0 → training starts from the published weights | structural | removes the source of the gap |
| 2 | Parameter groups | 18 adult query rows + text LoRA at 0.1× LR; the new 9 rows + context modules at full LR | one config | medium–high |
| 3 | Selection rule | select on `min(adult18, peds23)` — but **only on validation**; because that makes adult-18 a selected quantity, the gate is measured on the **test split** (§17) | one function + a test split | medium |
| 4 | Per-cohort loss mask | cuts the pediatric gradient of the rare adult classes (Arterial, Coronary, Emphysema, Hiatal) | low | small — at most ~0.0018, all four inside their CIs; the rationale is hygiene, not the gate |
| 5 | Batch composition | a fixed peds:adult ratio (e.g. 1:3) instead of the natural 14:86 | low | medium |
| 6 | CT-RATE training masks | routing switches on for the adult half too (86% currently runs maskless) | ~1,740 GPU-hours | unknown — validation evidence first |
| 7 | Weight averaging (soup) | blend the adapted model with `arcct_seed0.pt` | minutes | low but almost free |

> **the gate is not met today — and the "within the noise band" defence is invalid**
>
> ```
> # PAIRED patient-level bootstrap · 3,002 volumes · 1,304 patients · 1,000 replicates
> ARC-CT (augmd_seed0) macro-18 = 0.8574   # reproduced exactly against FINAL_REPORT
>
> C47harm  = 0.8442    difference = +0.0132   90% CI [0.0110, 0.0154]
> C47mix   = 0.8400    difference = +0.0174   90% CI [0.0145, 0.0203]
> P(ours ≥ ARC-CT) = 0.000
> ```
>
> **This is now a paired test.** ARC-CT's per-volume predictions were already sitting in
> `results/adult_gate/augmd_seed0/predictions.npz` (18 August), over the same 3,002 volumes
> in the same order — so no extra run was needed. Pairing cut the interval to a third
> (half-width 0.0061 → **0.0022**), because the sampling noise comes from the same volumes
> in both models and largely cancels. The gap is now known precisely: **0.0132 ± 0.0022**.
>
> Rev.4 said here that "if the difference ≤ the seed band it is indistinguishable". That
> makes three separate errors at once: **(a)** overlapping confidence intervals are *not*
> an equivalence test — failing to reject a difference is not evidence of equivalence;
> **(b)** the margin is being chosen after seeing the data; **(c)** it treated the reference
> as having no interval of its own — whereas the predictions were on disk and the paired
> test is available today (above).
>
> **The correct criterion — a pre-registered TOST:** non-regression may be declared only if
> the upper bound of the 90% CI on the paired difference falls below a margin Δ declared
> *before* the runs. And the choice of margin is **decisive** here: today's upper bound is
> **0.0154**, so with Δ = 0.01 it is **not equivalent**, and with Δ = 0.02 it **is**. The
> only difference between the two conclusions is the margin itself — which is why a Δ
> defended on clinical or literature grounds must be fixed before the results are seen.
> **Status today: at Δ = 0.01 the gate is not met.**

`14 · mandatory`

## Ablation ladder

The study cited in the notes found that on 3D chest CT, simple concatenation beat
cross-attention, and attributed that to insufficient data. Our pediatric training set is
7,042 volumes (the training portion of the 8,816 cohort) — and the adult half does not
compensate, because CT-RATE has an indication in 49.8% of cases with a median of 2 words.
The ladder is not optional.

- **R1 · CT only** — baseline · today's C47 line
- **R2 · CT + concat(indication)** — the cheapest fusion · the winner in the cited study · if we cannot beat this, cross-attention is unjustified
- **R3 · + cross-attention conditioning (C1, no FiLM)** — the spatial question · the first rung that uses the token sequence
- **R4 · + FiLM channel gating (C1 full)** — "which kind of feature" · completes the two-level conditioning
- **R5 · + relevance weighting (C2)** — w = 1 + βr · pool and loss together
- **R6 · fusion: concat / gated sum** — Z = W[Z_gen;Z_ind] vs Z_gen + g⊙Z_ind
- **R7 · isolation: masked / shared self-attention** — a direct measurement of the leak · if the shared version scores better, the reason is the leak
- **R8 · age: band / scalar / flat table / shuffled** — the rationale for the ordinal-cumulative parameterisation

> **arms deferred to Phase 2**
>
> C3 (the learned λ gate) · C4 (free indication queries) · peds mask-free / no anatomy
> queries · cropping × gate 2×2. All four remain as real arms, but they are not on
> Phase 1's delivery list — the reasons are in §07 and §08.
>
> **Ordering, not cancellation.** All four can enter the queue the moment Phase 1's
> *measurement* closes; GPUs are not the constraint, and the arms are independent and
> parallelisable. If Phase 1 clears its bar, C3/C4 warm-start from its checkpoint. If it
> does not, the same arms run as rescue arms — C3 being the closest candidate to raising
> pediatric AUC, since peds is the cohort most exposed to the empty-region cliff (§07).
> In that case the loss of mask-free inference is reported explicitly alongside the
> headline metric.

### Four evaluation conditions

| input | purpose | pre-declared expectation |
|---|---|---|
| CT only | baseline | current performance |
| CT + correct indication | expected clinical use | highest |
| CT + no indication | robustness | close to baseline, not below it |
| CT + **mismatched** indication | **indication-bias test** | the nodule prediction must not disappear — if it does, the conditioning is too strong |

*The fourth row tests the mechanism's reason for existing. The example from the notes:
the CT contains a pulmonary nodule and the indication says "rule out pneumonia". The
nodule prediction must survive.*

> **rev.5 · the statistical floor, rewritten from scratch**
>
> **0.0028 cannot be used as a decision threshold.** It comes from a single seed pair:
> σ̂ = |d|/√2 = 0.0020, with **1 degree of freedom**. The upper end of a two-sided 95%
> interval on a 1-df variance estimate is 0.063 (the one-sided 95% limit is 0.032) — i.e.
> 32× the point estimate. Worse, the row that reports the band contradicts it: in the same
> column V1 = 0.8005 and V1-seed1 = 0.7933 sit side by side, and the reader computes
> 0.0072. (The two were measured on different label sets — the band comes from 0.7961 vs
> 0.7933 — but that is not visible in the table.)
>
> **And it can never be used per class.** The measured per-class seed bands range from
> 0.0001 to 0.0500. Rev.4's §10 inference that "Lung nodule −0.0080, it got worse" was
> already uninterpretable, because that class's own band is 0.0118.
>
> **Rev.5 rules:**
>
> - Before the ablation begins, **3 seeds** are run on a single configuration and σ is measured with usable degrees of freedom.
> - Every number is given with a **patient-level bootstrap CI**; per-class differences are judged against **per-class CIs**, never against the mean-level band.
> - Power: resolving Δ = 0.005 at α = 0.05 with 80% power needs **3 seeds** per arm if σ = 0.002 and **16** if σ = 0.005; with Bonferroni over the 24 macro comparisons, 6 and 32. **We do not know which σ is right** — which is the rationale for the 3-seed pre-measurement.
> - **Multiplicity:** 8 rungs × 3 cohorts = **24** macro comparisons → Holm–Bonferroni; 8 × 27 = **216** class comparisons → Benjamini–Hochberg FDR (q = 0.10). The pre-registered predictions are analysed separately and uncorrected.
> - **Budget:** 3 seeds × 10 configurations ≈ **450 GPU-hours** (750 at 5 seeds). Rev.4 budgeted nothing for this — while it is **26%** of the CT-RATE masks (1,740) that were deferred as "too expensive". In rev.7 this fell from 675 once C3/C4 moved to Phase 2. **But GPUs are not the binding constraint on this project** and arms run multi-GPU in parallel: going to 5 seeds enlarges the queue, not the wall clock. So 3 seeds is not a budget concession — it is what the σ pre-measurement returns; if σ comes out at 0.005, nothing about the resources blocks going to 16 seeds.
> - **The 9.19% figure belongs to the wrong artefact:** it was measured on the *region* cache (88,170 volume-region pairs), not on the 27-class pathology labels. The run-to-run instability of the pathology labels has **never been measured** → added to Phase 0.

`15 · the headline visual`

## The key experiment: same CT, three indications

The experiment flagged in the notes as a "much more interesting contribution". The ARC-CT
paper already showed that Grad-CAM is more focal than CT-CLIP's; this is the dynamic
version of that.

> **[FIGURE 4]** Proposed experiment (not yet run). The image does not change; only the
> clinical question does. The attention mass of the conditioned pathology queries
> redistributes according to the question, while the unconditioned representation stays
> fixed. This single figure shows both that the mechanism works and that it is safe.
> `Z_gen`'s equality across the three conditions is not a *hope* here but a structural
> consequence of the isolation in §04 — the experiment does not confirm it, it *audits*
> that it has not been violated. What is actually measured is that the class ranking
> changes meaningfully.

`16 · what could go wrong`

## Risks and controls

> **from the notes · the risk to take most seriously**
>
> *"The model could learn: indication says 'PE' → predict PE, rather than actually finding
> an embolus. This is probably the biggest danger of this approach."*

| risk | status | control |
|---|---|---|
| Indication shortcut (multimodal shortcut) | the biggest danger | **Four layers:** 30% dropout · mismatched-indication test · counterfactual consistency loss (over the prediction logits, §12) · the image-free indication-only baseline. All four are reported. |
| The indication *suppressing* a finding | prevented by architecture | residual conditioning (q̃ = q + Δ) · w ≥ 1 · the unconditioned bank always open |
| Indication → label leakage (text overlap) | `[measured]` indication keywords in 46 volumes (0.52%); `LEAKAGE_EXCLUDE.txt` holds **54** rows with a safety margin (46 indication + 9 age, 1 both) | the 54 volumes are excluded from **every arm's** validation (not just the indication arm — otherwise the denominators do not match) |
| Indication → label leakage (semantic) | `[open]` — the 27 definitions were written by reading pediatric reports | the shuffle test · keyword scanning cannot see this |
| Cohort-flag leakage | prevented by design | the model is never given a site/cohort indicator — only age |
| PHI in the indication text | `[open]` — median 21 words, contains names and dates | scrubbing before anything reaches the model + IRB · the procedure is documented |
| The label pipeline is not deterministic | `[partly]` 9.19% **on the region cache**; the pathology labels were never measured (Phase 0/9) | label snapshots are frozen · seed controls mandatory |
| Dual-bank cost | accepted, ~1.82× Q-Former attention | the dual bank introduces no new query parameters (weight sharing); C4 + the clinical-question query add 5 new rows · the cost is reported |

`17 · order of work`

## Phased plan

All of Phase 0 is done without writing model code. Items 1–2 determine the "mask-free
peds" decision, item 3 determines where C1 attaches. **Item 5 was run and refuted its
prediction** (§10); 8–9 establish the statistical floor.

| # | measurement | what it needs | time | what it decides |
|---|---|---|---|---|
| 1 | Adult routing penalty | 3,000 masks already on disk — one line in the eval script | ~30 min | is the penalty specific to pediatrics · **the mask-free peds decision** |
| 2 | Penalty ↔ checkpoint step | intermediate checkpoints saved every 400/800 steps | ~1 h | can selection bias be eliminated · **the mask-free peds decision** |
| 3 | The R2 readout head | the 9 existing checkpoints | ~1 h | is there a free gain · **where C1 attaches** |
| 4 | Age × region survival breakdown | 8,816 masks + age metadata | ~1 h | C3's opening figure · the rationale for the λ prediction |
| 5 | Padding fraction × per-class AUC — **RUN** | existing checkpoints | done | **prediction refuted** (§10) — cropping stays in Phase 2, its expected gain has fallen |
| 6 | Indication-only baseline (image-free) | text + age | ~4 h | the significance floor for the entire indication arm |
| 7 | "Interval change" language in the reports | one extraction pass | ~1 day | settles the longitudinal arm (4,813 pairs) on its own |
| 8 | 3 seeds, one configuration | the existing peds fine-tune recipe | ~45 h GPU | measure σ with usable degrees of freedom — how many seeds the ladder needs depends on it |
| 9 | Pathology label re-extraction | the same prompt, a second pass | ~3 h GPU | run-to-run instability of the 27-class labels (9.19% is the region cache's figure, not this one) |

*Items 1, 2 and 3 together take ~3 hours and determine whether the sentence "hard masking
hurts" can go into the paper at all.*

### Phase 1 · the minimum publishable paper

- **C1 + C2** in full (cross-attn + FiLM + relevance), dual bank, concat fusion.
- **Empty-region cliff fix** (a bug fix, independent of C3): ρ is computed before the override and the empty-region rate is logged.
- Age band + sex as context tokens; `L_ind` + `L_loc`.
- **Isolation of the unconditioned path**: self-attn group mask, λ split, group-restricted pooling, fusion W = [I ; 0] — and a bit-identity assertion before training.
- **A third, never-touched test split** (patient-level). Selection and all ablation ranking happen on validation; the headline pediatric and adult numbers are produced **once**, on the test split, after the configuration is frozen. Today the selection metric and the reported metric are the same — with a maximum taken over ~30–60 checkpoints (an upper bound under an independence assumption), the optimism is on the order of **~0.0065**, i.e. half the adult gap under discussion.
- **Radiologist adjudication**: ~200 pediatric validation volumes, two pediatric radiologists, 27 classes. Today **no** label has been verified by a human; moreover the labels and the contrastive text target derive from the same report, so their noise is correlated and the AUCs are optimistic relative to an independent reference.
- **Age-stratified reporting**: every pediatric number three times — pooled, <18 (n=1,280), ≥18 (n=492). The abstract quotes <18. Otherwise "it works in children" cannot be separated from "it works in the adult quarter of a children's hospital".
- Per-cohort loss mask; parameter groups; `min(adult18, peds23)` selection.
- Three-cohort evaluation + four conditions + the eight-rung ladder.

### Phase 2 · strengthening

- **C3 — the learned λ gate**, if Phase 0/1–3 validates the rationale and the inference-mask problem is solved (§07).
- **C4 — 4 free indication queries**, if the mismatched-indication test passes cleanly on C1+C2 (§08).
- **Longitudinal delta queries** — 4,813 pairs, 1,390 patients. If Phase 0/7 supports it.
- **Structured indication concepts** (symptom / clinical context / target condition / anatomy) as separate query tokens — the notes' "multiple queries can independently interrogate the CT" idea.
- **Body-normalised cropping** (Dr. Kurugol's directive, §10) — Phase 0/5 determines its order, not its fate. Implementation (B): box = 6 integers per volume, ~0 disk (§10).
- **CT-RATE training masks** (~1,740 GPU-hours) — only if Phase 0/1–2 justifies it.
- **Publishing the pediatric benchmark protocol**: the data is PHI, so the schema + prompt + protocol + model are published, not the data.

`18 · decisions`

## Open decisions

The notes closed five of the previous eight questions. What remains, and what has newly opened:

1. **Phase 0 first?** Three measurements, three hours, determining both the "mask-free peds" decision and the paper's opening claim. My strong recommendation: yes.
2. **Do we accept the dual-bank + isolation cost?** (1.72× in Phase 1, 1.82× if C4 arrives.) The unconditioned pathology path is the strictest reading of the notes' "unconditional pathway" requirement: 1.82× Q-Former cross-attention, plus a self-attention mask parameter on `QFormerBlock` and splitting λ in two. The cheaper alternative is a single bank + residual conditioning + w ≥ 1 — but then the protection drops to two layers and `Z_gen`'s independence from the indication stops being structural. My preference is the dual bank: §15's headline measurable claim (`Z_gen` identical under three conditions) cannot be made with a single bank, and it is the first thing a reviewer will look at.
3. **Pneumothorax:** should it stay, with the region-8 fix, as C3's declared test case? (I do not recommend removing it from the model; rationale in §11.)
4. **Arterial wall calcification:** your suggestion was right and the decision changed — loss-masked in peds, absent from the peds score. The pediatric positives (ligamentum arteriosum, catheter tracts, cartilage, veins) are not the adult disease. One correction to the expectation only: we measured it, the adult gain is about −0.0052, so this will not close the gate. The definition fix's 0.654→0.735 gain is still reported — as a *label-quality* finding, not a training decision.
5. **If it is not re-initialisation, what caused the adult loss?** This question *opened* in rev.5 rather than closing: all 18 rows started bit-identical and were still lost. The three classes carrying measurable loss (Mosaic, Fibrotic sequela, Lung nodule) are simultaneously on the "age-dependent meaning" and the "annotator threshold mismatch" lists — and label harmonisation alone moves Mosaic by 0.0287 (32% of the −0.0803) and Interlobular by 0.0759. **The discriminating experiment:** a 2×2 of {harmonised / published labels} × {fixed step budget}, ≥2 seeds per cell, early stopping off — 8 runs, ~120 GPU-hours. Do we run this before Phase 1?
6. **Is the pediatric headline metric "27" or "measurable 23"?** Dropping four classes (Emphysema, Coronary, Hiatal hernia, Arterial wall calc) takes 27 → 23. For V2: 27-class 0.7974 · 23-class 0.7944. The difference is about one seed band. **The decision rule must be disease identity, not the number**, and it must be fixed now, before the results are seen.
7. **CT-RATE training masks** (~1,740 GPU-hours) — tied to the outcome of Phase 0/1, or committed now?
8. **IRB + PHI scrubbing** for the indication text — who, and when? This is C1's only genuine external dependency.
9. **Citation verification.** The notes cite **two** works: Dia-LLaMA (2025, disease-aware attention) and the unnamed study finding concat > cross-attention on 3D chest CT. **Di Piazza et al. does not appear in the notes** — that comes from our own `CONTEXT_ROUTED_CT_PROPOSAL.md` and was misattributed to the notes in rev.4.

---

*Sources: `reports/Dr_Kurugols_notes.txt` (25 KB, 26 August) · `reports/IMPORTANT.md` ·
`reports/ARC_CT_PEDIATRIC_REPORT.md` · `reports/CONTEXT_ROUTED_CT_PROPOSAL.md` ·
`arc-ct@9c04648` + local changes · `eval_matrix/` (V1–V5, C16, C47mix, C47harm) ·
8 evaluation runs dated 25 August 2026. Code readings: `anatomy_qformer.py`, `qformer.py`,
`train_stage2.py`, `evaluate.py`, `schema.py`.*

---

## Revision history

**Rev. 2** — rewritten after Dr. Kurugol's notes arrived: dual pathology bank,
token-sequence cross-attention, relevance gate, free indication queries, split global
queries, the ablation ladder, the resolution section. Dropped: the age-conditioned
prevalence prior (demoted to optional).

**Rev. 3** — four defects found in a design audit were closed:
(i) self-attention was leaking into the unconditioned bank → group mask;
(ii) the λ gate tied the attention geometry of the unconditioned queries to the indication
→ the λ^gen / λ^ind split;
(iii) the fusion `W` started randomly → [I ; 0];
(iv) the dot-product form of `r_c` was untrainable with frozen towers → MLP.
The pool was also normalised (division by Σw), β was made learned and 0-init, and the
counterfactual consistency loss was removed — isolation appeared to make it unnecessary.

**Rev. 4** — the adult gap was measured per class for the first time; *Arterial wall
calcification* was removed from pediatric training and scoring; the headline metric became
"measurable 23".

**Rev. 5** — after a six-lane independent audit (numbers · code claims · architecture ·
internal consistency · fidelity to the notes · statistics). **Two claims were retracted:**
(1) §13's "5 re-initialised query rows" causality — a checkpoint measurement showed that no
pathology row was ever re-initialised; (2) §10's "Lung nodule got worse" claim — the sign
was inverted, the correct value is +0.0021.
**Design errors closed:** β's runaway solution (softplus), the empty-region guard
neutralising λ, ρ being read inverted, **a fourth leakage path (pooling)**, the `H_dem`
dimension mismatch, `L_loc` working against C1, which bank `L_ind` supervises, and γ being
able to erase the query.
**Restored:** the counterfactual consistency loss — but over the prediction logits, not
`Z_gen`. **Added:** the TOST equivalence criterion, the reportability floor, the test split,
radiologist adjudication, age-stratified reporting, power/multiplicity/budget.

**Rev. 5, second audit round** (numbers + consistency, run on rev.5). Five *retraction
survivors* were closed — the document was contradicting its own retraction in five places.
**A third correction:** rev.5 had compared per-class deltas against the marginal interval of
a single arm and concluded "15/18 are inside the noise, no share can be reported"; because
the two models are measured on the same volumes the right tool is a **paired** bootstrap,
and there **12/18 exclude zero**. Computing a share is legitimate: the three classes whose
meaning is age-dependent carry **61%** of the gross loss. Also: ARC-CT's predictions were
found to be *already on disk* (the Phase-0 item was removed), the paired difference was
measured at **+0.0132 [0.0110, 0.0154]**, the cropping × class-AUC test was run and
**refuted**, and ~12 numeric corrections were made.

**Rev. 6** — the document was read end to end and **24 inconsistencies** were closed. The
most serious: §13 still said "the reference has no interval of its own" (while two
paragraphs above it stated the predictions were on disk); in §10 the old paragraph of the
refuted hypothesis was sitting *below* its refutation; §05 said "the mask restriction
remains exactly as it is" while §07 softened it; §03's H2 hypothesis was wrong (selection
is not mask-free but oracle-masked); four rows of the §02 table contradicted the body; and
the region-survival numbers came from the abandoned 13-region scheme (10%/3% →
**5.6%/1.5%**).

**Rev. 7** — the scope was narrowed. **C1 + C2 are the whole of Phase 1**; C3 and C4 were
moved to optional arms. Rationale: C3's motivation rests on a measurement that §03 Fact 2
shows is confounded, *and* through ρ it would require a mask at inference, risking ARC-CT's
mask-free-inference property; C4 is the largest shortcut surface in the design and its
"contributes nothing when the indication is absent" claim is not mechanised. Closing the
empty-region cliff was separated from C3 and became a **Phase 1 bug fix**. Result: the
ladder 12 → **8**, macro comparisons 36 → **24**, budget 675 → **~450 GPU-hours**, the query
bank 71 → **67 slots**, cost 1.82× → **1.72×**.
