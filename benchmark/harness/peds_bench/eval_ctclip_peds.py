#!/usr/bin/env python3
"""Zero-shot pediatric evaluation of CT-CLIP, replicating scripts/zero_shot.py.

CT-CLIP's own readout (zero_shot.py:133-143) is, per pathology:
    text = [f"{p} is present.", f"{p} is not present."]
    output = model(text_tokens, volume, device=device)   # cosine sim * temp
    output = softmax(output, dim=0); take index 0
We call the model exactly that way. Unlike GreenRFM, CT-CLIP L2-normalises both
latents inside forward (ct_clip.py:~730) and multiplies by a learned temperature,
so this readout is well-posed for cross-volume ranking.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/ch278233/BENCHMARK/CT-CLIP/scripts")
sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")

CACHE = "/temp_work/ch278233/PEDS_BENCH/ctclip_npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vollist", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt")
    ap.add_argument("--labels", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--naive", action="store_true",
                    help="one forward per class, exactly as zero_shot.py loops (23x slower)")
    ap.add_argument("--verify", type=int, default=0,
                    help="check fast path == naive path on this many volumes, then continue")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    from transformer_maskgit import CTViT
    from transformers import BertTokenizer, BertModel
    from ct_clip import CTCLIP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    te = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    ie = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
               temporal_patch_size=10, spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=ie, text_encoder=te, dim_text=768, dim_image=294912,
                  dim_latent=512, extra_latent_projection=False, use_mlm=False,
                  downsample_image_embeds=False, use_all_token_embeds=False)
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    own = clip.state_dict()
    shared = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    print(f"[ctclip-eval] {os.path.basename(a.ckpt)}: {len(shared)}/{len(own)} tensors match", flush=True)
    if len(shared) < 0.9 * len(own):
        sys.exit("[ctclip-eval] FATAL: architecture mismatch")
    clip.load_state_dict(sd, strict=False)
    clip = clip.to(device).eval()

    ldf = pd.read_csv(a.labels)
    classes = [c for c in ldf.columns if c != "VolumeName"]
    lab = {r["VolumeName"]: np.array([r[c] for c in classes], dtype=np.float32)
           for _, r in ldf.iterrows()}
    vols = [l.strip() for l in open(a.vollist) if l.strip()]

    # Pre-tokenise the 23 present/absent prompt pairs once.
    prompts = [[f"{c} is present.", f"{c} is not present."] for c in classes]
    toks = [tok(p, return_tensors="pt", padding="max_length", truncation=True,
                max_length=512).to(device) for p in prompts]
    # Flat 46-prompt tokenisation for the fast path.
    flat = [t for pair in prompts for t in pair]
    tok_all = tok(flat, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=512).to(device)

    def score_naive(vol):
        """Exactly zero_shot.py:133-143 -- one full forward per class."""
        return [float(torch.softmax(clip(tk, vol, device=device), dim=0)[0].item())
                for tk in toks]

    def score_fast(vol):
        """Identical arithmetic, image encoded once instead of 23 times.

        CT-CLIP's forward L2-normalises both latents and scores with
        einsum('b d, b d -> b', text, image) * temperature.exp() (ct_clip.py:806).
        With return_latents we get those same normalised latents and reproduce the
        dot product directly, so this is the same number, not an approximation.
        """
        tl, il, _ = clip(tok_all, vol, return_latents=True, device=device)
        temp = clip.temperature.exp()
        sims = (il @ tl.t()).squeeze(0) * temp          # (46,)
        return torch.softmax(sims.view(len(classes), 2), dim=1)[:, 0].tolist()

    score = score_naive if a.naive else score_fast

    # Resume support: a 35-minute GPU job on a preemptable partition can be killed
    # partway through, so checkpoint progress and skip what is already done.
    part = a.out.replace(".npz", "_partial.npz")
    preds, trues, used = [], [], []
    done = set()
    if os.path.exists(part):
        z = np.load(part, allow_pickle=True)
        preds = [list(r) for r in z["pred"]]
        trues = [r for r in z["true"]]
        used = [str(x) for x in z["volumes"]]
        done = set(used)
        print(f"[ctclip-eval] resuming: {len(done)} volumes already scored", flush=True)
    miss = 0

    def save_partial():
        np.savez_compressed(part + ".tmp.npz",
                            pred=np.asarray(preds, dtype=np.float32),
                            true=np.asarray(trues, dtype=np.int32),
                            classes=np.array(classes), volumes=np.array(used))
        os.replace(part + ".tmp.npz", part)
    with torch.no_grad():
        for i, v in enumerate(vols):
            stem = v[:-7] if v.endswith(".nii.gz") else v
            f = os.path.join(a.cache, stem + ".npz")
            if stem in done:
                continue
            if v not in lab or not os.path.exists(f):
                miss += 1
                continue
            arr = np.load(f)["arr"].astype(np.float32)
            vol = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,240,480,480)
            if a.verify and len(preds) < a.verify:
                rf, rn = score_fast(vol), score_naive(vol)
                dmax = max(abs(x - y) for x, y in zip(rf, rn))
                print(f"[verify] {stem}: max|hizli-naive| = {dmax:.3e}", flush=True)
                if dmax > 1e-4:
                    sys.exit(f"[verify] FATAL: paths disagree by {dmax:.3e} -- not equivalent")
            row = score(vol)
            preds.append(row)
            trues.append(lab[v])
            used.append(stem)
            if (i + 1) % 25 == 0:
                save_partial()
                print(f"[ctclip-eval] {i+1}/{len(vols)} (kullanilan={len(preds)})", flush=True)
            if a.limit and len(preds) >= a.limit:
                break

    if not preds:
        sys.exit("[ctclip-eval] FATAL: no volumes evaluated")
    pred = np.asarray(preds, dtype=np.float32)
    true = np.asarray(trues, dtype=np.int32)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, pred=pred, true=true, classes=np.array(classes),
                        volumes=np.array(used))
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(true[:, j], pred[:, j]) for j in range(len(classes))
            if len(np.unique(true[:, j])) > 1]
    print(f"[ctclip-eval] n={len(pred)} eksik={miss} macroAUC={np.mean(aucs):.4f} "
          f"({len(aucs)}/{len(classes)} sinif) -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
