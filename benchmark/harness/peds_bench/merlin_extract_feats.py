#!/usr/bin/env python
"""
Merlin frozen-feature extraction for the pediatric chest-CT benchmark.

STREAMING: each volume is loaded, transformed with Merlin's OWN MONAI chain,
encoded by the frozen image encoder, reduced to a 2048-d vector, and the
volume is discarded. No preprocessed volume is ever written to disk.

Merlin is an ABDOMINAL/pelvic CT foundation model. Applying it to chest CT is
an out-of-domain transfer -- see the report caveat.
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch

def log(*a):
    print(*a, flush=True)

def load_map():
    m = {}
    with open("/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m[row["volume_name"]] = row["nii_path"]
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "valid"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="/temp_work/ch278233/PEDS_BENCH/merlin/feats")
    args = ap.parse_args()

    from merlin.data.monai_transforms import ImageTransforms
    from merlin.models.build import MerlinArchitecture

    vollist = f"/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_{args.split}.txt"
    vols = [l.strip() for l in open(vollist) if l.strip()]
    if args.limit:
        vols = vols[: args.limit]
    vmap = load_map()
    vols = [v for v in vols if v in vmap]
    shard_vols = vols[args.shard :: args.nshards]
    log(f"[shard {args.shard}/{args.nshards}] split={args.split} n={len(shard_vols)} of {len(vols)}")

    # ---- build frozen Merlin, load checkpoint, report tensor match ----
    CKPT = ("/temp_work/ch278233/PEDS_BENCH/weights/merlin/"
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt")
    model = MerlinArchitecture(ImageEmbedding=True)   # Merlin's own code, unmodified
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    msd = model.state_dict()
    matched = [k for k in sd if k in msd and msd[k].shape == sd[k].shape]
    shape_mismatch = [k for k in sd if k in msd and msd[k].shape != sd[k].shape]
    missing = [k for k in msd if k not in sd]
    unexpected = [k for k in sd if k not in msd]
    res = model.load_state_dict(sd, strict=True)
    log(f"[ckpt] ckpt_tensors={len(sd)} model_tensors={len(msd)} "
        f"matched={len(matched)} shape_mismatch={len(shape_mismatch)} "
        f"missing={len(missing)} unexpected={len(unexpected)} strict_load=OK")
    if args.shard == 0:
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"ckpt_tensors": len(sd), "model_tensors": len(msd),
                   "matched": len(matched), "shape_mismatch": len(shape_mismatch),
                   "missing": missing, "unexpected": unexpected,
                   "image_encoder_tensors": sum(1 for k in matched if k.startswith("encode_image"))},
                  open(os.path.join(args.outdir, "ckpt_match.json"), "w"), indent=2)

    enc = model.encode_image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    # ---- streaming dataset: transform in workers, never persisted ----
    class DS(torch.utils.data.Dataset):
        def __init__(self, names):
            self.names = names
        def __len__(self):
            return len(self.names)
        def __getitem__(self, i):
            name = self.names[i]
            try:
                out = ImageTransforms({"image": vmap[name]})
                img = out["image"]
                if hasattr(img, "as_tensor"):
                    img = img.as_tensor()
                return i, img.float(), True
            except Exception as e:
                sys.stderr.write(f"FAIL {name}: {type(e).__name__}: {e}\n")
                return i, torch.zeros(1, 224, 224, 160), False

    dl = torch.utils.data.DataLoader(
        DS(shard_vols), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=lambda b: b, prefetch_factor=2 if args.workers else None)

    feats = np.zeros((len(shard_vols), 2048), dtype=np.float32)
    ok = np.zeros(len(shard_vols), dtype=bool)
    failures = []
    t0 = time.time(); done = 0
    with torch.no_grad():
        for batch in dl:
            good = [(i, im) for i, im, g in batch if g]
            for i, im, g in batch:
                if not g:
                    failures.append(shard_vols[i])
            if not good:
                done += len(batch); continue
            idxs = [i for i, _ in good]
            x = torch.stack([im for _, im in good]).to(dev, non_blocking=True)
            try:
                f = enc(x)
                f = f.reshape(-1, 2048)          # encoder returns (1,B,2048)
                assert f.shape[0] == len(idxs), f"got {f.shape} for {len(idxs)}"
                feats[idxs] = f.float().cpu().numpy()
                ok[idxs] = True
            except Exception as e:
                sys.stderr.write(f"GPUFAIL {[shard_vols[i] for i in idxs]}: {e}\n")
                failures.extend(shard_vols[i] for i in idxs)
            done += len(batch)
            if done % 20 < args.batch_size:
                el = time.time() - t0
                log(f"  {done}/{len(shard_vols)}  {el/max(done,1):.2f}s/vol  eta={(len(shard_vols)-done)*el/max(done,1)/60:.1f}min")
            del x
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.split}_shard{args.shard:02d}.npz")
    np.savez_compressed(out, feats=feats[ok], vols=np.array(shard_vols, dtype=object)[ok].astype(str),
                        failed=np.array(failures, dtype=str))
    log(f"[done] wrote {out}  ok={int(ok.sum())}/{len(shard_vols)} failed={len(failures)}")
    if failures:
        log("[failed] " + ", ".join(failures))

if __name__ == "__main__":
    main()
