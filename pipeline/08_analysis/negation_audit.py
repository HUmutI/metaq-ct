#!/usr/bin/env python3
"""How often does a labeller cite a sentence that NEGATES the label it asserts?

The citation audit catches labels smeared onto an unrelated sentence. This
catches the opposite failure: the cited sentence is exactly on topic, and says
the finding is absent. "MEDIASTINUM: No mediastinal lymphadenopathy." cited as
evidence FOR Lymphadenopathy is not a judgement call, it is a reading error,
and unlike a definition dispute it needs no gold standard to score.

Reuses kw_baseline's clause-scoped negation (window bounded by [.;:*] or
but/however/although, plus the forward "is not identified" form) so the rule
here is the same one already unit-tested there, not a second ad-hoc regex.

Claude's gold-sample citations run through the identical check as a control.
"""
from __future__ import annotations
import csv, glob, json, re, sys
from collections import defaultdict

sys.path.insert(0, "/home/ch278233/pipeline/08_analysis")
from kw_baseline import _suppressed, DEFAULT
from citation_audit import RX, EV, GOLD, CLA, qwen_pairs, claude_pairs


def negated_citation(lab: str, sent: str) -> bool:
    """True if every mention of the finding in the cited sentence is negated."""
    rx = RX.get(lab)
    if rx is None or not sent:
        return False
    hits = list(rx.finditer(sent))
    if not hits:
        return False                       # no anchor: the other audit's case
    return all(_suppressed(sent, m.start(), m.end(), DEFAULT) == "negated"
               for m in hits)


def run(pairs):
    st = defaultdict(lambda: [0, 0])
    for lab, sent in pairs:
        if lab not in RX:
            continue
        st[lab][0] += 1
        if negated_citation(lab, sent):
            st[lab][1] += 1
    return st


def main():
    q, c = run(qwen_pairs()), run(claude_pairs())
    print("positives whose cited sentence negates the finding\n")
    print("%-38s %8s %7s   %8s %7s" % ("class", "Qwen n", "neg", "Cla n", "neg"))
    print("-" * 76)
    rows, qn, qu, cn, cu = [], 0, 0, 0, 0
    for lab in RX:
        a, b = q.get(lab, [0, 0]), c.get(lab, [0, 0])
        rows.append((a[1] / a[0] if a[0] else 0, lab, a, b))
        qn += a[0]; qu += a[1]; cn += b[0]; cu += b[1]
    for _, lab, a, b in sorted(rows, key=lambda r: -r[0]):
        if a[1] == 0 and b[1] == 0:
            continue
        qs = "%6.1f%%" % (100 * a[1] / a[0]) if a[0] else "     -"
        cs = "%6.1f%%" % (100 * b[1] / b[0]) if b[0] else "     -"
        print("%-38s %8d %7s   %8d %7s" % (lab, a[0], qs, b[0], cs))
    print("-" * 76)
    print("%-38s %8d %6.2f%%   %8d %6.2f%%"
          % ("TOTAL (all classes)", qn, 100 * qu / max(qn, 1),
             cn, 100 * cu / max(cn, 1)))

    print("\n--- examples (Qwen) ---")
    shown = defaultdict(int)
    for lab, sent in qwen_pairs():
        if shown[lab] < 2 and negated_citation(lab, sent):
            shown[lab] += 1
            print("  %-34s %s" % (lab, (sent or "")[:110]))
        if sum(shown.values()) >= 14:
            break


if __name__ == "__main__":
    main()
