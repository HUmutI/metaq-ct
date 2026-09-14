#!/usr/bin/env python3
"""Cache CT-CLIP image latents once so prompt templates can be swept for free.

CT-CLIP's zero-shot readout is softmax over a (positive, negative) prompt pair of
cosine similarities between L2-normalised latents scaled by a learned temperature
(ct_clip.py:806). The image half of that does not depend on the prompt, yet
zero_shot.py re-encodes the volume for every class AND would re-encode it again
for every template you wanted to try. Encoding once and storing the 512-d latent
makes a template sweep instant and costs ~3 MB for 1507 volumes.

Nothing about the model changes: latents come straight from the model's own
forward(return_latents=True) path.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/ch278233/BENCHMARK/CT-CLIP/scripts")
CACHE = "/temp_work/ch278233/PEDS_BENCH/ctclip_npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vollist", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt")
    ap.add_argument("--labels", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv")
    ap.add_argument("--cache", default=CACHE)
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
    print(f"[latents] {os.path.basename(a.ckpt)}: {len(shared)}/{len(own)} tensors match", flush=True)
    if len(shared) < 0.9 * len(own):
        sys.exit("[latents] FATAL: architecture mismatch")
    clip.load_state_dict(sd, strict=False)
    clip = clip.to(device).eval()

    ldf = pd.read_csv(a.labels)
    classes = [c for c in ldf.columns if c != "VolumeName"]
    lab = {r["VolumeName"]: np.array([r[c] for c in classes], dtype=np.float32)
           for _, r in ldf.iterrows()}
    vols = [l.strip() for l in open(a.vollist) if l.strip()]

    part = a.out.replace(".npz", "_partial.npz")
    lats, trues, used = [], [], []
    if os.path.exists(part):
        z = np.load(part, allow_pickle=True)
        lats = [r for r in z["latents"]]; trues = [r for r in z["true"]]
        used = [str(x) for x in z["volumes"]]
        print(f"[latents] resuming at {len(used)}", flush=True)
    done = set(used)

    def save(path):
        np.savez_compressed(path + ".tmp.npz",
                            latents=np.asarray(lats, dtype=np.float32),
                            true=np.asarray(trues, dtype=np.int32),
                            classes=np.array(classes), volumes=np.array(used),
                            temperature=np.float32(clip.temperature.exp().item()))
        os.replace(path + ".tmp.npz", path)

    with torch.no_grad():
        # One dummy text is required only because forward() takes both towers;
        # the returned image latent does not depend on it.
        dummy = tok(["x"], return_tensors="pt", padding="max_length",
                    truncation=True, max_length=512).to(device)
        for i, v in enumerate(vols):
            stem = v[:-7] if v.endswith(".nii.gz") else v
            f = os.path.join(a.cache, stem + ".npz")
            if stem in done or v not in lab or not os.path.exists(f):
                continue
            arr = np.load(f)["arr"].astype(np.float32)
            vol = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            _tl, il, _ = clip(dummy, vol, return_latents=True, device=device)
            lats.append(il.squeeze(0).float().cpu().numpy())
            trues.append(lab[v]); used.append(stem)
            if (i + 1) % 50 == 0:
                save(part); print(f"[latents] {i+1}/{len(vols)} (n={len(lats)})", flush=True)

    if not lats:
        sys.exit("[latents] FATAL: nothing encoded")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save(a.out)
    L = np.asarray(lats)
    print(f"[latents] n={L.shape[0]} dim={L.shape[1]} temp={clip.temperature.exp().item():.4f} -> {a.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
