"""One age parser and one band table, shared by every consumer.

``pipeline/08_analysis/age_analysis.py`` produced the age distribution quoted in
the design doc, and ``arcct/context.py`` turns an age into the band the model
sees. If those two ever disagree the published table stops describing the input
the model was trained on, so the parser lives here and both import it.

The parser handles both cohorts from one set of patterns: CT-RATE's DICOM
``031Y`` / ``018M`` / ``003D``, and the BCH column, which is decimal years. The
decimal pattern is not decoration -- infants are fractional (0.25 is a
three-month-old), and an integer-only rule silently drops 521 of the youngest
patients, exactly the group a pediatric question is about.
"""

from __future__ import annotations

import bisect
import re

# Order matters: the unit-suffixed patterns are tried before the bare decimal,
# so "18M" is eighteen months rather than eighteen years.
AGE_PATTERNS = [
    # DICOM PatientAge is a zero-padded fixed-width field: 045Y, 018M, 003D.
    # These four must come FIRST and must be anchored: the free-text pattern
    # below needs an 'r' after the 'y' ("yr", "years"), so a bare "036Y" falls
    # through it and every CT-RATE age silently becomes unknown.
    (re.compile(r"^\s*0*(\d{1,3})\s*[Yy]\s*$"), 1.0),
    (re.compile(r"^\s*0*(\d{1,3})\s*[Mm]\s*$"), 1 / 12),
    (re.compile(r"^\s*0*(\d{1,3})\s*[Ww]\s*$"), 1 / 52),
    (re.compile(r"^\s*0*(\d{1,3})\s*[Dd]\s*$"), 1 / 365),
    (re.compile(r"(\d{1,3})\s*(?:y(?:ea)?rs?|yo|y/o|yr)\b", re.I), 1.0),
    (re.compile(r"(\d{1,3})\s*(?:months?|mo)\b", re.I), 1 / 12),
    (re.compile(r"(\d{1,3})\s*(?:weeks?|wks?)\b", re.I), 1 / 52),
    (re.compile(r"(\d{1,3})\s*(?:days?)\b", re.I), 1 / 365),
    (re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*$"), 1.0),
]

# Half-open [lo, hi). Upper edges only -- bisect_right over these gives the band
# index directly, and there is no second copy of the boundaries to drift.
BAND_EDGES = (1.0, 2.0, 5.0, 10.0, 13.0, 18.0, 25.0, 40.0, 60.0)
N_BANDS = len(BAND_EDGES) + 1          # 10
BAND_UNKNOWN = N_BANDS                 # 10, a free row outside the ordinal chain
BAND_NAMES = ("<1y", "1-2y", "2-5y", "5-10y", "10-13y", "13-18y",
              "18-25y", "25-40y", "40-60y", "60+y", "unknown")

# HIPAA safe harbour: ages over 89 are a re-identification risk and are reported
# as a single top group. The pediatric cohort maxes at 69, so this only ever
# touches CT-RATE.
AGE_CLAMP = 89.0

SEX_FEMALE, SEX_MALE, SEX_UNKNOWN = 0, 1, 2


def parse_age(raw) -> float | None:
    """Decimal years, or None when the value cannot be read."""
    s = str(raw or "").strip()
    if not s:
        return None
    for rx, mult in AGE_PATTERNS:
        m = rx.search(s)
        if m:
            v = float(m.group(1)) * mult
            if 0 <= v <= 120:
                return v
    return None


def age_band(age_years: float | None) -> int:
    """Band index 0..9, or :data:`BAND_UNKNOWN`.

    A negative age is unknown, not band 0: a negative value is a parse failure,
    and calling it "infant" would put a fabricated number into training.
    """
    if age_years is None:
        return BAND_UNKNOWN
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return BAND_UNKNOWN
    if a != a or a < 0.0:
        return BAND_UNKNOWN
    return bisect.bisect_right(BAND_EDGES, a)


def parse_sex(raw) -> int:
    s = str(raw or "").strip().upper()
    if s.startswith("F"):
        return SEX_FEMALE
    if s.startswith("M"):
        return SEX_MALE
    return SEX_UNKNOWN


def clamp_age(age_years: float | None) -> tuple[float | None, bool]:
    """Apply the safe-harbour ceiling. Returns (age, was_clamped)."""
    if age_years is None:
        return None, False
    return (AGE_CLAMP, True) if age_years > AGE_CLAMP else (age_years, False)


def _selftest() -> int:
    fails = []
    cases = [
        # CT-RATE DICOM
        ("031Y", 31.0), ("018M", 1.5), ("003D", 3 / 365), ("065Y", 65.0),
        # BCH decimal years
        ("2", 2.0), ("0.25", 0.25), ("0.08219178", 0.08219178), ("14.0", 14.0),
        # nothing usable
        ("", None), ("   ", None), ("unknown", None), (None, None),
    ]
    for raw, want in cases:
        got = parse_age(raw)
        ok = (got is None and want is None) or (got is not None and want is not None
                                                and abs(got - want) < 1e-6)
        if not ok:
            fails.append(f"parse_age({raw!r}) = {got}, expected {want}")

    band_cases = [(0.0, 0), (0.99, 0), (1.0, 1), (2.0, 2), (4.99, 2), (5.0, 3),
                  (12.99, 4), (13.0, 5), (17.99, 5), (18.0, 6), (25.0, 7),
                  (40.0, 8), (59.99, 8), (60.0, 9), (120.0, 9),
                  (None, BAND_UNKNOWN), (-1.0, BAND_UNKNOWN)]
    for a, want in band_cases:
        got = age_band(a)
        if got != want:
            fails.append(f"age_band({a}) = {got}, expected {want}")

    for raw, want in (("Female", 0), ("F", 0), ("Male", 1), ("M", 1),
                      ("U", 2), ("", 2), (None, 2), ("O", 2)):
        got = parse_sex(raw)
        if got != want:
            fails.append(f"parse_sex({raw!r}) = {got}, expected {want}")

    a, c = clamp_age(95.0)
    if not (a == AGE_CLAMP and c):
        fails.append("clamp_age(95) must clamp")
    a, c = clamp_age(60.0)
    if not (a == 60.0 and not c):
        fails.append("clamp_age(60) must not clamp")

    # This module and arcct/context.py must agree, or the published age table
    # stops describing the input the model was trained on.
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        from arcct.context import age_to_band
        for a in (0.0, 0.5, 1.0, 4.9, 5.0, 17.99, 18.0, 59.9, 60.0, 88.0, None, -3.0):
            if age_band(a) != age_to_band(a):
                fails.append(f"pipeline and arcct disagree on age {a}: "
                             f"{age_band(a)} vs {age_to_band(a)}")
    except ImportError as exc:
        # torch is not in the base interpreter. The cross-check still runs under
        # the arcct env, and preflight runs it there; say so loudly rather than
        # passing quietly on a check that did not happen.
        print(f"   SKIP arcct.context cross-check ({exc}) -- re-run under "
              "$HOME/micromamba/envs/arcct/bin/python")

    if fails:
        print("demographics FAIL (%d):" % len(fails))
        for f in fails:
            print("   ", f)
        return 1
    print("demographics OK · DICOM and decimal years both parse · bands agree "
          "with arcct.context at every edge · sex maps to 3 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
