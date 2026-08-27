"""Slot layout for the indication-conditioned Q-Former bank.

ARC-CT's ``AnatomyQFormer`` holds one learnable parameter whose shape depends on
the label schema: ``qformer.queries`` of shape ``(n_queries, dim)``, laid out as
anatomy, then pathology, then global. Phase 1 keeps that parameter and widens
only the *attention bank* around it.

The bank has 67 attention slots but only 40 distinct parameter rows, because the
27 conditioned pathology slots are the 27 general rows read a second time:

    slot   0..9    anatomy              -> param rows  0..9    (unchanged)
    slot  10..36   pathology, general   -> param rows 10..36   (unchanged)
    slot  37..38   global, general      -> param rows 37..38   (unchanged)
    ------------------------------------ n_gen = 39 boundary --
    slot  39..65   pathology, CONDITIONED -> param rows 10..36 (WEIGHT-TIED)
    slot     66    global, clinical     -> param row  39       (the ONE new row)

Unconditioned-first and contiguous is not a cosmetic choice; four things depend
on it.

1.  ``queries[:39]`` is byte-identical to today's parameter, so an ARC-CT
    checkpoint loads by appending a single row rather than by remapping.
2.  ``gen`` and ``cond`` are contiguous slices, so the self-attention group
    constraint is two slice-and-concatenate attention calls rather than a
    67x67 additive mask, and ``Z_gen`` is ``tokens[:, :39].mean(1)``.
3.  ``build_role_mask`` rows 0..38 keep their present meaning; extending the
    mask to 67 rows is one ``index_select``.
4.  ``train_stage2.py``'s ``qf_tokens[:, A_q:A_q + P_q]`` still resolves to
    ``[:, 10:37]`` and still selects the general bank. That is arithmetic luck,
    not design -- use :attr:`SlotLayout.path_gen_index` instead, so a future
    layout change breaks loudly rather than silently re-slicing.

``slot_to_row`` and ``role_row`` are two different vectors and must never be
conflated: slot 66 draws its *parameters* from new row 39 but inherits its
*attention mask* from row 37, one of the unrestricted global rows. They agree
everywhere except that one position.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotLayout:
    """Immutable description of the query bank.

    Attributes are derived, never stored twice: everything below is a function
    of the four counts, so a layout cannot be internally inconsistent.
    """

    n_anatomy: int = 10
    n_path: int = 27
    n_glob_gen: int = 2
    n_glob_clin: int = 1
    n_path_cond: int = 27

    @classmethod
    def phase1(cls, n_path: int = 27, n_anatomy: int = 10, n_glob_gen: int = 2) -> "SlotLayout":
        """The Phase 1 bank: the conditioned pathology block mirrors the general one."""
        return cls(n_anatomy=n_anatomy, n_path=n_path, n_glob_gen=n_glob_gen,
                   n_glob_clin=1, n_path_cond=n_path)

    @classmethod
    def legacy(cls, n_path: int = 27, n_anatomy: int = 10, n_glob_gen: int = 2) -> "SlotLayout":
        """Today's ARC-CT bank: no conditioned block, no clinical query.

        Useful as the reference side of the step-0 identity test, and as the
        layout a context-free run degenerates to.
        """
        return cls(n_anatomy=n_anatomy, n_path=n_path, n_glob_gen=n_glob_gen,
                   n_glob_clin=0, n_path_cond=0)

    # -- counts ---------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Distinct parameter rows -- what ``qformer.queries`` actually holds."""
        return self.n_anatomy + self.n_path + self.n_glob_gen + self.n_glob_clin

    @property
    def n_gen(self) -> int:
        """Size of the unconditioned group. ``Z_gen`` averages exactly these."""
        return self.n_anatomy + self.n_path + self.n_glob_gen

    @property
    def n_cond(self) -> int:
        return self.n_path_cond + self.n_glob_clin

    @property
    def n_slots(self) -> int:
        """Attention positions in the bank."""
        return self.n_gen + self.n_cond

    @property
    def n_base_rows(self) -> int:
        """Rows ``build_role_mask`` produces: it knows nothing about conditioning."""
        return self.n_gen

    # -- index vectors --------------------------------------------------------
    # Plain lists, so this module never imports torch and can be unit-tested
    # without a GPU environment. Callers turn them into buffers.

    @property
    def gen_index(self) -> list[int]:
        return list(range(self.n_gen))

    @property
    def path_gen_index(self) -> list[int]:
        a = self.n_anatomy
        return list(range(a, a + self.n_path))

    @property
    def glob_gen_index(self) -> list[int]:
        s = self.n_anatomy + self.n_path
        return list(range(s, s + self.n_glob_gen))

    @property
    def cond_index(self) -> list[int]:
        return list(range(self.n_gen, self.n_slots))

    @property
    def path_cond_index(self) -> list[int]:
        return list(range(self.n_gen, self.n_gen + self.n_path_cond))

    @property
    def glob_clin_index(self) -> list[int]:
        s = self.n_gen + self.n_path_cond
        return list(range(s, s + self.n_glob_clin))

    @property
    def slot_to_row(self) -> list[int]:
        """slot -> parameter row. The conditioned block repeats the general rows."""
        out = list(range(self.n_gen))                       # anatomy + path + glob_gen
        out += self.path_gen_index[: self.n_path_cond]      # tied: same rows again
        out += [self.n_gen + i for i in range(self.n_glob_clin)]   # the new row(s)
        return out

    @property
    def role_row(self) -> list[int]:
        """slot -> row of the 39-row role mask it inherits.

        The conditioned pathology slots take the same organ union as their tied
        partners. The clinical global takes the first general global row, which
        ``build_role_mask`` has already set to all-True.
        """
        out = list(range(self.n_gen))
        out += self.path_gen_index[: self.n_path_cond]
        out += [self.glob_gen_index[0]] * self.n_glob_clin
        return out

    # -- invariants -----------------------------------------------------------

    def validate(self) -> None:
        """Raise on any layout that would silently mis-address the bank."""
        problems: list[str] = []
        if self.n_path_cond not in (0, self.n_path):
            problems.append(
                f"n_path_cond={self.n_path_cond} must be 0 or n_path={self.n_path}; "
                "a partial conditioned block has no defined tying")
        for name, v in (("slot_to_row", self.slot_to_row), ("role_row", self.role_row)):
            if len(v) != self.n_slots:
                problems.append(f"{name} has {len(v)} entries, expected {self.n_slots}")
        if max(self.slot_to_row, default=-1) >= self.n_rows:
            problems.append("slot_to_row addresses a parameter row that does not exist")
        if max(self.role_row, default=-1) >= self.n_base_rows:
            problems.append("role_row addresses a mask row build_role_mask never produces")
        if set(self.slot_to_row) != set(range(self.n_rows)):
            problems.append("some parameter row is never used by any slot -- it would "
                            "receive no gradient and drift as dead weight")
        # The one position where the two vectors are allowed to disagree is the
        # clinical global: new parameters, inherited mask.
        differ = [i for i, (a, b) in enumerate(zip(self.slot_to_row, self.role_row)) if a != b]
        if differ != self.glob_clin_index:
            problems.append(f"slot_to_row and role_row differ at {differ}, "
                            f"expected exactly {self.glob_clin_index}")
        if problems:
            raise ValueError("invalid SlotLayout:\n  " + "\n  ".join(problems))

    def describe(self) -> str:
        return (f"SlotLayout(slots={self.n_slots}, rows={self.n_rows}, gen={self.n_gen}, "
                f"anatomy={self.n_anatomy}, path={self.n_path}, "
                f"path_cond={self.n_path_cond}, glob_gen={self.n_glob_gen}, "
                f"glob_clin={self.n_glob_clin})")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


@dataclass(frozen=True)
class CtxConfig:
    """Every Phase 1 knob, read once from the environment.

    Defaults are chosen so that an unset environment reproduces today's run:
    ``enabled`` is False, and nothing else is consulted until it is True.
    """

    enabled: bool = False
    c1: bool = True                 # indication -> query conditioning
    c2: bool = True                 # relevance weighting of the conditioned pool
    film_eps: float = 0.2           # gamma = 1 + eps*tanh(.), bounded away from 0
    beta_init: float = -6.0         # b, with beta = softplus(b) >= 0 structurally
    ind_dropout: float = 0.3
    cf_prob: float = 0.25
    selfattn: str = "split"         # "split" | "mask" | "none" (ablation)
    zind_combine: str = "mean"      # "mean" | "add"
    fusion: str = "concat"          # "concat" | "gated"
    age_mode: str = "band"          # "band" | "scalar" | "flat" | "shuffled"
    ptok_weight: float = 0.5
    max_ind_len: int = 64
    role_stats: bool = True
    assert_invariance: bool = False
    debug: bool = False

    @classmethod
    def from_env(cls) -> "CtxConfig":
        cfg = cls(
            enabled=_env_flag("RAC_USE_CONTEXT_QFORMER", "0"),
            c1=_env_flag("RAC_CTX_C1", "1"),
            c2=_env_flag("RAC_CTX_C2", "1"),
            film_eps=float(os.environ.get("RAC_CTX_FILM_EPS", "0.2")),
            beta_init=float(os.environ.get("RAC_CTX_BETA_INIT", "-6.0")),
            ind_dropout=float(os.environ.get("RAC_IND_DROPOUT", "0.3")),
            cf_prob=float(os.environ.get("RAC_CF_PROB", "0.25")),
            selfattn=os.environ.get("RAC_CTX_SELFATTN", "split").lower(),
            zind_combine=os.environ.get("RAC_CTX_ZIND_COMBINE", "mean").lower(),
            fusion=os.environ.get("RAC_CTX_FUSION", "concat").lower(),
            age_mode=os.environ.get("RAC_AGE_MODE", "band").lower(),
            ptok_weight=float(os.environ.get("RAC_CTX_PTOK_WEIGHT", "0.5")),
            max_ind_len=int(os.environ.get("RAC_CTX_MAX_IND_LEN", "64")),
            role_stats=_env_flag("RAC_CTX_ROLE_STATS", "1"),
            assert_invariance=_env_flag("RAC_CTX_ASSERT_INVARIANCE", "0"),
            debug=_env_flag("RAC_CTX_DEBUG", "0"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems: list[str] = []
        # Every one of these rejects an unknown value rather than falling back
        # to the default. An ablation arm whose flag is silently ignored is not
        # an ablation arm, it is a duplicate of the baseline wearing its name --
        # and it would be written up as evidence.
        if self.selfattn not in ("split", "mask", "none"):
            problems.append(f"RAC_CTX_SELFATTN={self.selfattn!r}, expected "
                            "split|mask|none")
        if self.fusion not in ("concat", "gated"):
            problems.append(f"RAC_CTX_FUSION={self.fusion!r}, expected concat|gated")
        if self.age_mode not in ("band", "scalar", "flat", "shuffled"):
            problems.append(f"RAC_AGE_MODE={self.age_mode!r}, expected "
                            "band|scalar|flat|shuffled")
        if self.zind_combine not in ("mean", "add"):
            problems.append(f"RAC_CTX_ZIND_COMBINE={self.zind_combine!r}, expected mean|add")
        if not 0.0 < self.film_eps <= 1.0:
            # eps <= 0 disables FiLM silently; eps > 1 lets gamma reach 0 and the
            # "residual, the query is never erased" guarantee stops being true.
            problems.append(f"RAC_CTX_FILM_EPS={self.film_eps}, expected 0 < eps <= 1")
        if not 0.0 <= self.ind_dropout < 1.0:
            problems.append(f"RAC_IND_DROPOUT={self.ind_dropout}, expected [0, 1)")
        if not 0.0 <= self.cf_prob <= 1.0:
            problems.append(f"RAC_CF_PROB={self.cf_prob}, expected [0, 1]")
        if self.max_ind_len < 8:
            problems.append(f"RAC_CTX_MAX_IND_LEN={self.max_ind_len} is shorter than a "
                            "typical indication (measured median 24 words)")
        if problems:
            raise ValueError("invalid CtxConfig:\n  " + "\n  ".join(problems))


def _selftest() -> int:
    """Run as ``python -m arcct.slots``. No torch, no GPU, no data."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    L = SlotLayout.phase1(27)
    L.validate()
    check(L.n_slots == 67, f"n_slots={L.n_slots}, expected 67")
    check(L.n_rows == 40, f"n_rows={L.n_rows}, expected 40")
    check(L.n_gen == 39, f"n_gen={L.n_gen}, expected 39")
    check(L.n_cond == 28, f"n_cond={L.n_cond}, expected 28")

    check(L.gen_index == list(range(39)), "gen_index")
    check(L.path_gen_index == list(range(10, 37)), "path_gen_index")
    check(L.glob_gen_index == [37, 38], "glob_gen_index")
    check(L.cond_index == list(range(39, 67)), "cond_index")
    check(L.path_cond_index == list(range(39, 66)), "path_cond_index")
    check(L.glob_clin_index == [66], "glob_clin_index")

    s2r, rr = L.slot_to_row, L.role_row
    check(s2r[:39] == list(range(39)), "slot_to_row prefix must be the identity")
    check(s2r[39:66] == list(range(10, 37)), "conditioned block must reuse rows 10..36")
    check(s2r[66] == 39, "clinical global must use the new row 39")
    check(rr[66] == 37, "clinical global must inherit mask row 37")

    differ = [i for i, (a, b) in enumerate(zip(s2r, rr)) if a != b]
    check(differ == [66], f"slot_to_row and role_row differ at {differ}, expected [66]")

    # Tying is what makes rows < slots: rows 10..36 must each appear twice.
    from collections import Counter
    c = Counter(s2r)
    check(all(c[r] == 2 for r in range(10, 37)),
          "each general pathology row must back exactly two slots")
    check(all(c[r] == 1 for r in list(range(10)) + [37, 38, 39]),
          "every other row must back exactly one slot")
    check(sum(c.values()) == 67 and len(c) == 40, "row/slot accounting")

    # The legacy layout must be today's bank, and must survive validate().
    G = SlotLayout.legacy(27)
    G.validate()
    check(G.n_slots == 39 and G.n_rows == 39 and G.n_gen == 39, "legacy is 39/39/39")
    check(G.slot_to_row == G.role_row == list(range(39)),
          "legacy layout must be the identity on both vectors")

    # An 18-class adult bank must work too, with no hardcoded 27 anywhere.
    A = SlotLayout.phase1(18)
    A.validate()
    check((A.n_slots, A.n_rows, A.n_gen) == (49, 31, 30), f"18-class bank {A.describe()}")

    # Two clinical globals is a legitimate layout (both new rows are used), so
    # validate() must accept it -- the check below is about unused rows and
    # partial tying, not about the count being exactly one.
    two = SlotLayout(n_anatomy=10, n_path=27, n_glob_gen=2, n_glob_clin=2, n_path_cond=27)
    two.validate()
    check((two.n_slots, two.n_rows) == (68, 41), f"two-clinical bank {two.describe()}")

    # A partial conditioned block has no defined tying and must be rejected.
    bad = SlotLayout(n_anatomy=10, n_path=27, n_glob_gen=2, n_glob_clin=1, n_path_cond=5)
    try:
        bad.validate()
        fails.append("a partial conditioned block (n_path_cond=5) must be rejected")
    except ValueError:
        pass

    # CtxConfig
    cfg = CtxConfig.from_env()
    check(cfg.enabled is False, "context must be OFF unless RAC_USE_CONTEXT_QFORMER=1")
    check(cfg.selfattn == "split", "split is the default self-attention path")
    check(cfg.zind_combine == "mean", "mean is the default Z_ind combination")
    check(cfg.fusion == "concat" and cfg.age_mode == "band",
          "concat fusion and banded age are the defaults")
    for name, value in (("RAC_CTX_SELFATTN", "bogus"), ("RAC_CTX_FILM_EPS", "0"),
                        ("RAC_CTX_ZIND_COMBINE", "concat"), ("RAC_IND_DROPOUT", "1.0"),
                        ("RAC_CTX_FUSION", "sum"), ("RAC_AGE_MODE", "years")):
        old = os.environ.get(name)
        os.environ[name] = value
        try:
            CtxConfig.from_env()
            fails.append(f"{name}={value} must be rejected")
        except ValueError:
            pass
        finally:
            if old is None:
                del os.environ[name]
            else:
                os.environ[name] = old

    if fails:
        print("S1 FAIL (%d):" % len(fails))
        for f in fails:
            print("   ", f)
        return 1
    print("S1 OK ·", L.describe())
    print("        legacy:", G.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
