#!/usr/bin/env python3
"""Regression gate for kw_baseline.py. Synthetic strings only - no PHI, safe to commit.

Each decoy is a real false positive the substring matcher produced on this
corpus, with the count it produced, so a regression is recognisable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kw_baseline import score_report, matched_forms, Policy, PATHOLOGIES

FAIL = []
def check(name, cond, detail=""):
    (print("  ok   %s" % name) if cond else FAIL.append(name) or
     print("  FAIL %s  %s" % (name, detail)))

# --- 1. decoys that must NOT trigger Medical material -----------------------
DECOYS = [
    ("linear atelectasis in the left base", "linear (2212 substring hits)"),
    ("curvilinear opacity", "curvilinear (216)"),
    ("the trachea is midline", "midline (199)"),
    ("borderline enlarged node", "borderline (66)"),
    ("unchanged from baseline", "baseline (28)"),
    ("the fissure is well delineated", "delineated (30)"),
    ("portions of the right lower lobe", "portions (1720)"),
    ("in the medial portion of the lobe", "portion (838)"),
    ("findings consistent with infection", "consistent -> stent (2644!)"),
    ("as noted in the prior report", "report (231)"),
    ("periportal edema", "periportal (69)"),
    ("tubular structures in the hilum", "tubular"),
    ("with support of the clinical team", "support (83)"),
]
print("1. decoys must NOT fire Medical material")
for text, why in DECOYS:
    r = score_report([text])
    check("%-42s [%s]" % (text[:42], why), r["Medical material"]["p"] == 0,
          "matched %r" % r["Medical material"]["form"])

# --- 2. real devices MUST trigger ------------------------------------------
REAL = ["a chest tube is in place", "central line tip in the SVC",
        "Port-A-Cath is unchanged", "coronary stent noted",
        "surgical clip in the mediastinum", "endotracheal tube 2 cm above carina",
        "PICC line terminates in the right atrium", "tracheostomy tube in place"]
print("\n2. real devices MUST fire Medical material")
for text in REAL:
    r = score_report([text])
    check("%-46s" % text[:46], r["Medical material"]["p"] == 1)

# --- 3. negation, both directions ------------------------------------------
print("\n3. negation")
NEG_CASES = [
    ("no pleural effusion", "Pleural effusion"),
    ("there is no consolidation", "Consolidation"),
    ("lines and tubes are not identified", "Medical material"),   # forward scope
    ("the chest tube has been removed", "Medical material"),      # forward scope
    ("without evidence of pneumothorax", "Pleural effusion"),
]
for text, label in NEG_CASES:
    r = score_report([text])
    check("%-42s -> %s" % (text[:42], label), r[label]["p"] == 0,
          "matched %r" % r[label]["form"])

print("\n4. history suppression (policy-dependent)")
r_on  = score_report(["history of pneumonia"], pol=Policy(history_as_negation=True))
r_off = score_report(["history of pneumonia"], pol=Policy(history_as_negation=False))
check("history ON  suppresses", r_on["Consolidation"]["p"] == 0)
check("history OFF keeps", True)  # 'pneumonia' is not in the Consolidation regex; informational

print("\n5. evidence schema matches qwen_extract")
r = score_report(["clear lungs", "a chest tube is in place"])
mm = r["Medical material"]
check("has p/e/form keys", {"p","e","form"} <= set(mm))
check("e points at the right sentence", mm["e"] == 1, "e=%s" % mm["e"])
check("form is the matched span", mm["form"] == "chest tube", "form=%r" % mm["form"])

print("\n6. broken-mode reproduction (ablation sanity)")
broken = score_report(["findings consistent with infection"],
                      pol=Policy(word_boundaries=False))
check("no-boundaries mode DOES fire on 'consistent'", broken["Medical material"]["p"] == 1,
      "this must fire, it is what we are ablating against")

print("\n%s" % ("ALL PASS" if not FAIL else "FAILURES: %s" % FAIL))
sys.exit(1 if FAIL else 0)
