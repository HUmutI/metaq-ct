#!/usr/bin/env python3
"""Write the overnight result to a file, so it survives a dropped connection.

Everything on the critical path is chained inside SLURM and does not need a live
session. What a dropped VPN costs is not the work but the reading of it: nobody
is left to collect the numbers. So the last job in the chain writes them down.

Compares the V2 pediatric fine-tune against V1 class by class. V1 scored 0.7961
mean over 27 classes, dragged there by labels rather than by the model: on the
18 classes shared with the adult schema it reached 0.8205 against the adult
model's 0.8574, a gap of only 0.037, while "Arterial wall calcification" alone
lost 0.298 because the pediatric extraction had quietly relabelled it as
calcified catheter tracts in veins.
"""
from __future__ import annotations
import csv, glob, json, os, subprocess, sys, time

OUT = "/temp_work/ch278233/SABAH_RAPORU.txt"
V1_RES = "/temp_work/ch278233/runs/peds_finetune"
V2_RES = "/temp_work/ch278233/runs/peds_finetune_v2"
V3_RES = "/temp_work/ch278233/runs/peds_finetune_v3"
CTL_RES = "/temp_work/ch278233/runs/peds_finetune_v1_seed1"
V4_RES = "/temp_work/ch278233/runs/peds_finetune_v4"
V5_RES = "/temp_work/ch278233/runs/peds_finetune_v5"


def load_auc(d):
    fs = sorted(glob.glob(os.path.join(d, "best_auc_update*.txt")),
                key=lambda p: int(p.rsplit("update", 1)[1].split(".")[0]))
    if not fs:
        return {}, None
    out = {}
    for ln in open(fs[-1]):
        if ":" in ln:
            k, v = ln.rsplit(":", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out, os.path.basename(fs[-1])


def prevalence(path):
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return {}
    cols = [c for c in rows[0] if c != "VolumeName"] if rows else []
    return {c: 100.0 * sum(float(r[c] or 0) > 0.5 for r in rows) / len(rows)
            for c in cols}


def main() -> int:
    L = []
    L.append("=" * 78)
    L.append("SABAH RAPORU   %s" % time.strftime("%Y-%m-%d %H:%M"))
    L.append("=" * 78)

    # Auto-discover every finished run instead of naming them one by one.
    # Adding V4 and V5 by hand meant twice forgetting to update this file and
    # the watchdog alongside the new job; a directory scan cannot forget.
    L.append("\nBULUNAN KOSULAR")
    for d in sorted(glob.glob("/temp_work/ch278233/runs/*")):
        a, f = load_auc(d)
        if a:
            L.append("  %-34s %-28s ort %.4f"
                     % (os.path.basename(d), f, a.get("Mean AUC", float("nan"))))

    v1, f1 = load_auc(V1_RES)
    v2, f2 = load_auc(V2_RES)
    v3, f3 = load_auc(V3_RES)
    vc, fc = load_auc(CTL_RES)
    v4, f4 = load_auc(V4_RES)
    v5, f5 = load_auc(V5_RES)
    L.append("\nV1 eski etiketler             : %s" % (f1 or "yok"))
    L.append("V2 yeni etiketler             : %s" % (f2 or "yok"))
    L.append("V3 yeni etiketler + bos-bolge : %s" % (f3 or "yok"))
    L.append("V4 + top-K odaksal havuzlama   : %s" % (f4 or "yok"))
    L.append("V5 + cok olcekli izgara (24^3)  : %s" % (f5 or "yok"))
    if v5 and v3:
        L.append("\nV5 - V3, cok olcekli izgaranin tek basina etkisi:")
        for k in ("Lung nodule", "Bone lesion or fracture",
                  "Pleural thickening or nodule", "Pulmonary metastases", "Mean AUC"):
            if k in v3 and k in v5:
                L.append("  %-42s %.4f -> %.4f  %+.4f" % (k, v3[k], v5[k], v5[k] - v3[k]))
    L.append("KONTROL V1 etiketleri, tohum 1 : %s" % (fc or "yok"))
    if v4 and v3:
        FOCAL = ["Lung nodule", "Atelectasis", "Lung opacity",
                 "Post-surgical or post-treatment change"]
        L.append("\nV4 - V3, top-K havuzlamanin tek basina etkisi (odaksal siniflar):")
        for k in FOCAL:
            if k in v3 and k in v4:
                L.append("  %-42s %.4f -> %.4f  %+.4f" % (k, v3[k], v4[k], v4[k] - v3[k]))
        if "Mean AUC" in v4 and "Mean AUC" in v3:
            L.append("  %-42s %.4f -> %.4f  %+.4f"
                     % ("ORTALAMA", v3["Mean AUC"], v4["Mean AUC"],
                        v4["Mean AUC"] - v3["Mean AUC"]))
    if vc and v1 and "Mean AUC" in vc and "Mean AUC" in v1:
        band = abs(vc["Mean AUC"] - v1["Mean AUC"])
        L.append("\nGURULTU BANDI (ayni etiket, farkli tohum): %.4f vs %.4f = %.4f"
                 % (v1["Mean AUC"], vc["Mean AUC"], band))
        if v2 and "Mean AUC" in v2:
            gain = v2["Mean AUC"] - v1["Mean AUC"]
            verdict = ("ETIKETLERE ATFEDILEBILIR" if abs(gain) > 2 * band
                       else "gurultu bandinin icinde - atfedilemez")
            L.append("V2 kazanci %+.4f  ->  %s" % (gain, verdict))

    if v2:
        L.append("\n%-42s %8s %8s %8s %9s" % ("sinif", "V1", "V2", "V3", "V2-V1"))
        L.append("-" * 80)
        names = [k for k in v2 if k != "Mean AUC"]
        for k in sorted(names, key=lambda k: v2[k] - v1.get(k, 0)):
            a = v1.get(k)
            c = v3.get(k)
            cs = "%8.4f" % c if c is not None else "%8s" % "-"
            if a is None:
                L.append("%-42s %8s %8.4f %s %9s" % (k, "-", v2[k], cs, "yeni"))
            else:
                L.append("%-42s %8.4f %8.4f %s %+9.4f" % (k, a, v2[k], cs, v2[k] - a))
        L.append("-" * 80)
        L.append("%-42s %8.4f %8.4f %8s %+9.4f"
                 % ("ORTALAMA (27)", v1.get("Mean AUC", 0), v2.get("Mean AUC", 0),
                    ("%.4f" % v3["Mean AUC"]) if "Mean AUC" in v3 else "-",
                    v2.get("Mean AUC", 0) - v1.get("Mean AUC", 0)))

        # the extra line the mean does not show: three classes have too few
        # positives in 1773 validation volumes to mean anything (Emphysema 2,
        # Coronary calcification 6, Hiatal hernia 18), and they inflate it.
        thin = ["Emphysema", "Coronary artery wall calcification", "Hiatal hernia"]
        ev = [k for k in names if k not in thin]
        if ev:
            L.append("%-42s %8.4f %8.4f %8s %+9.4f"
                     % ("ORTALAMA (olculebilir 24)",
                        sum(v1.get(k, 0) for k in ev) / len(ev),
                        sum(v2[k] for k in ev) / len(ev),
                        ("%.4f" % (sum(v3[k] for k in ev if k in v3) / len(ev)))
                        if all(k in v3 for k in ev) else "-",
                        sum(v2[k] - v1.get(k, 0) for k in ev) / len(ev)))

    # Per-class comparison against the seed band. The mean is a bad summary
    # here: one class of 27 improving by 0.08 moves it by 0.003, which is
    # exactly what happened with Arterial wall calcification.
    if v1 and vc and v2:
        names = [k for k in v1 if k != "Mean AUC"]
        L.append("\n%-42s %8s %8s %9s %8s" % ("sinif", "V1ort", "yeni", "kazanc", "band"))
        L.append("-" * 80)
        rows = []
        for k in names:
            if k not in vc or k not in v2:
                continue
            base = (v1[k] + vc[k]) / 2
            new_runs = [d[k] for d in (v2, v3) if k in d]
            newv = sum(new_runs) / len(new_runs)
            band = abs(v1[k] - vc[k])
            rows.append((newv - base, k, base, newv, band))
        for gain, k, base, newv, band in sorted(rows, reverse=True):
            sig = "  <<<" if band > 0 and abs(gain) > 2 * band else ""
            L.append("%-42s %8.4f %8.4f %+9.4f %8.4f%s" % (k, base, newv, gain, band, sig))
        L.append("-" * 80)
        L.append("UYARI: band, sinif basina TEK tohum ciftinden hesaplaniyor - yani")
        L.append("gurultunun 1 orneklik tahmini. 27 sinifta birkacinin bandi tesadufen")
        L.append("cok kucuk cikar ve '<<<' isareti sisirilir. Onceden tahmin edilmis bir")
        L.append("sinif (Arterial wall calcification) disinda bu isareti kanit sayma.")

    p1 = prevalence("/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv")
    p2 = prevalence("/temp_work/ch278233/BCH_DATASET/LABELS27_V2/labels.csv")
    if p1 and p2:
        L.append("\nEtiket prevalansi, en cok degisen 8 sinif (8817 pediatrik hacim)")
        L.append("%-42s %8s %8s %9s" % ("sinif", "V1%", "V2%", "fark"))
        for c in sorted(p2, key=lambda c: -abs(p2[c] - p1.get(c, 0)))[:8]:
            L.append("%-42s %7.2f%% %7.2f%% %+8.2f" % (c, p1.get(c, 0), p2[c], p2[c] - p1.get(c, 0)))

    pa = prevalence("/temp_work/ch278233/CTRATE/CTRATE_labels27.csv")
    if pa:
        NEW9 = ["Post-surgical or post-treatment change", "Pulmonary metastases",
                "Tree-in-bud", "Pulmonary cyst", "Mass or neoplasm", "Mucus plugging",
                "Pleural thickening or nodule", "Bone lesion or fracture", "Pneumothorax"]
        L.append("\n9 pediatrik sinifin GERCEK yetiskin prevalansi (onceki deger: hepsi 0.00%%)")
        for c in NEW9:
            if c in pa:
                L.append("  %-42s %6.2f%%" % (c, pa[c]))

    # Single-evaluator comparison on the pediatric validation split. This is the
    # number that answers "does adding adults help pediatrics", and it does not
    # come from the training logs: those are each model against its own labels,
    # and the combined run's own mean is on a 49%-adult set that is not
    # comparable to anything here.
    import glob as _g
    ev = {}
    for d in sorted(_g.glob("/temp_work/ch278233/eval_peds_only/*")):
        f = _g.glob(os.path.join(d, "*.txt"))
        if not f:
            continue
        for ln in open(f[0]):
            if ln.startswith("Mean AUC"):
                parts = ln.split(":")[1].split()
                ev[os.path.basename(d)] = (float(parts[0]), float(parts[1]))
                break
    if ev:
        L.append("\nPEDIATRIK-ONLY DEGERLENDIRME (1773 hacim, tek degerlendirici)")
        L.append("%-24s %12s %12s" % ("model/etiket", "maskesiz", "maske-yonl."))
        L.append("-" * 50)
        for k in sorted(ev):
            L.append("%-24s %12.4f %12.4f" % (k, ev[k][0], ev[k][1]))
        L.append("")
        L.append("Etiket kumesi notrl degil: '_lblv2' sonekli satirlar yeniden")
        L.append("cikarilan etiketlere karsi olculdu. Siralama iki kumede de ayni")
        L.append("cikarsa sonuc saglamdir.")

    try:
        q = subprocess.run(["squeue", "-u", "ch278233", "-o", "%.12i %.10j %.2t %.8M"],
                           capture_output=True, text=True, timeout=30).stdout
        L.append("\nKuyruk:\n%s" % q.rstrip())
    except Exception:
        pass

    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
