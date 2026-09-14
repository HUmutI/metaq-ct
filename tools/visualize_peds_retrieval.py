#!/usr/bin/env python3
"""Qualitative pediatric retrieval galleries, in the layout of Fig. 4 of
arXiv:2602.07872: a query on the left, then one row per method showing its
top-5 retrieved studies, each panel annotated with an objective relevance
number.

Two figures, same queries and same three methods, because the two tasks have
opposite winners and that is the point:
  image -> image  panels scored by label-set Jaccard against the query
  report -> image panels marked with the ground-truth study (green) and each
                  row annotated with the rank that study received

Query selection is method-agnostic and stated in the caption: among test
studies with exactly three positive findings, the two whose findings are
rarest in the cohort (lowest summed prevalence), so the cases are clinically
distinctive without being chosen for how any method scores on them.
"""
from __future__ import annotations
import argparse, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

WORK = "/temp_work/ch278233"
VOLROOT = f"{WORK}/PEDS_NPZ_NESTED"
E = f"{WORK}/eval_matrix/peds23_retrieval"
B = f"{WORK}/PEDS_BENCH/retrieval"
sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")

# (row label, latent file, image-key, text-key). Ours twice: the two readouts
# whose ordering flips between the two tasks.
METHODS = [
    (r"MetaQ-CT  $z_{\rm ind}$", f"{E}/refined_c1_seed0_conditioned/retrieval_latents_refined_c1_seed0_conditioned.npz"),
    (r"MetaQ-CT  $z_{\rm gen}$", f"{E}/ct_only_seed0_general/retrieval_latents_ct_only_seed0_general.npz"),
    ("MPS-CT",                   f"{B}/mpsct_valid.npz"),
]
TOPK = 5
# Intensity convention of the preprocessed volumes, measured from the data:
# air -3.02, lung parenchyma -1.15, soft tissue ~0, bone >0.7. This window is
# the analogue of a clinical lung window on that scale.
WIN = (-1.70, 0.35)


def stem(a: str) -> str:
    return str(a).replace(".nii.gz", "")


def load_methods():
    out = []
    for name, f in METHODS:
        d = np.load(f, allow_pickle=True)
        acc = [stem(a) for a in d["accessions"]]
        img = d["img_lat"].astype(np.float64)
        txt = d["txt_lat"].astype(np.float64)
        img /= np.linalg.norm(img, axis=1, keepdims=True) + 1e-12
        txt /= np.linalg.norm(txt, axis=1, keepdims=True) + 1e-12
        out.append({"name": name, "acc": acc, "img": img, "txt": txt})
    ref = out[0]["acc"]
    for m in out[1:]:
        assert m["acc"] == ref, "latent files are not in the same study order"
    return out, ref


def vol_path(s: str) -> str:
    return f"{VOLROOT}/{s.rsplit('_', 1)[0]}/{s}/{s}.npz"


def lung_profile(s: str):
    """Per-slice fraction of lung-density voxels in the central half of the field."""
    p = vol_path(s)
    if not os.path.exists(p):
        return None
    v = np.load(p)["arr_0"]
    z, h, w = v.shape
    c = v[:, h // 4:3 * h // 4, w // 4:3 * w // 4]
    f = ((c > -2.0) & (c < -0.80)).reshape(z, -1).mean(1)
    # Boundary slices carry partial-volume values that fall inside the lung band,
    # so a raw argmax lands on slice 0. Smooth, then search the central 70%.
    k = max(3, z // 20)
    f = np.convolve(f, np.ones(k) / k, mode="same")
    lo, hi = int(0.15 * z), max(int(0.85 * z), int(0.15 * z) + 1)
    g = np.zeros_like(f)
    g[lo:hi] = f[lo:hi]
    return g


def axial_slice(s: str):
    """One representative axial slice: the one with the most lung-density voxels
    inside the central half of the field, which lands at mid-thorax."""
    p = vol_path(s)
    if not os.path.exists(p):
        return None
    v = np.load(p)["arr_0"]
    return v[int(np.argmax(lung_profile(s)))]


def panel(ax, s, edge=None, lw=2.0):
    im = axial_slice(s)
    if im is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7)
    else:
        ax.imshow(np.flipud(im), cmap="gray", vmin=WIN[0], vmax=WIN[1], interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge is not None:
            sp.set_color(edge); sp.set_linewidth(lw)


def jaccard(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def pick_queries(labels, classes, accs, n=2, npos=3):
    prev = labels.sum(0)
    cand = []
    for i, row in enumerate(labels):
        if row.sum() != npos:
            continue
        lp = lung_profile(accs[i])
        # a study whose field of view does not actually frame the thorax makes a
        # poor visual query; this filter is independent of any method's scores
        if lp is None or lp.max() < 0.18:
            continue
        cand.append((float(prev[row.astype(bool)].sum()), i))
    cand.sort()
    return [i for _, i in cand[:n]]


def short(names, row, maxn=3):
    got = [names[j] for j in np.where(row.astype(bool))[0]]
    return got[:maxn]


def build(mode, methods, accs, labels, classes, queries, out):
    nq, nm = len(queries), len(methods)
    fig_h = 2.15 * nm * nq + 0.75
    fig = plt.figure(figsize=(13.2, fig_h))
    gs = fig.add_gridspec(nm * nq, TOPK + 2, width_ratios=[1.32, 0.13] + [1] * TOPK,
                          hspace=0.30, wspace=0.06,
                          left=0.005, right=0.995, top=1 - 0.30 / fig_h, bottom=0.32 / fig_h)
    for qi, q in enumerate(queries):
        base = qi * nm
        # query cell spans this block's method rows
        axq = fig.add_subplot(gs[base:base + nm, 0])
        if mode == "i2i":
            panel(axq, accs[q], edge="#1a7f37", lw=2.4)
            axq.set_xlabel("query study\n" + "\n".join(short(classes, labels[q])),
                           fontsize=8.0, labelpad=4)
        else:
            axq.remove()
            from peds_report_text import texts_for
            axt = fig.add_subplot(gs[base:base + nm - 1, 0]); axt.axis("off")
            raw = " ".join(texts_for([accs[q]])[0].split())
            lines = textwrap.wrap(raw, width=50)
            clipped = len(lines) > 15
            lines = lines[:15]
            if clipped:
                lines[-1] += " \u2026"
            axt.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=axt.transAxes,
                                    facecolor="#f5f5f3", edgecolor="#1a7f37", linewidth=1.4))
            axt.text(0.06, 0.95, "query report", transform=axt.transAxes, fontsize=8.2,
                     va="top", style="italic", color="#1a7f37")
            axt.text(0.06, 0.855, "\n".join(lines), transform=axt.transAxes, fontsize=6.5,
                     va="top", linespacing=1.55)
            axg = fig.add_subplot(gs[base + nm - 1, 0])
            panel(axg, accs[q], edge="#1a7f37", lw=2.2)
            axg.set_xlabel("ground-truth study for this report", fontsize=7.6, labelpad=3)
        for mi, m in enumerate(methods):
            r = base + mi
            if mode == "i2i":
                sim = m["img"] @ m["img"][q]
                sim[q] = -np.inf
                order = np.argsort(-sim)[:TOPK]
                notes = [f"{jaccard(labels[q], labels[j]):.2f}" for j in order]
                marks = [None] * TOPK
                rowtag = m["name"]
            else:
                sim = m["img"] @ m["txt"][q]
                full = np.argsort(-sim)
                order = full[:TOPK]
                rank = int(np.where(full == q)[0][0]) + 1
                notes = [""] * TOPK
                marks = ["#1a7f37" if j == q else None for j in order]
                rowtag = m["name"] + f"\nrank {rank}"
            for k, j in enumerate(order):
                ax = fig.add_subplot(gs[r, k + 2])
                panel(ax, accs[j], edge=marks[k] if mode == "r2i" else "#cccccc",
                      lw=2.4 if (mode == "r2i" and marks[k]) else 0.6)
                if notes[k]:
                    ax.set_xlabel(notes[k], fontsize=8.0, labelpad=2)
                if r == base and qi == 0 and mi == 0:
                    ax.set_title(f"rank {k+1}", fontsize=8.5, pad=3)
            axl = fig.add_subplot(gs[r, 1], frame_on=False)
            axl.set_xticks([]); axl.set_yticks([]); axl.patch.set_alpha(0)
            axl.text(0.5, 0.5, rowtag, transform=axl.transAxes, rotation=90,
                     ha="center", va="center", fontsize=8.0, linespacing=1.25)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/ch278233/paper/figures")
    a = ap.parse_args()
    methods, accs = load_methods()
    d = np.load(f"{B}/mpsct_valid.npz", allow_pickle=True)
    labels, classes = d["labels"].astype(bool), [str(c) for c in d["classes"]]
    queries = pick_queries(labels, classes, accs)
    print("queries:", [(accs[q], short(classes, labels[q])) for q in queries])
    build("i2i", methods, accs, labels, classes, queries, f"{a.out_dir}/peds_retrieval_i2i.png")
    build("r2i", methods, accs, labels, classes, queries, f"{a.out_dir}/peds_retrieval_r2i.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
