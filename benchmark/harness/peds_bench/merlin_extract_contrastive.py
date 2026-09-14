#!/usr/bin/env python
"""
Merlin CONTRASTIVE (text-aligned, 512-d) embeddings for the pediatric VALID split.

Uses Merlin's own image+text contrastive forward, unmodified:
    image_features, ehr_features, text_features = MerlinArchitecture()(image, [report,...])
which returns the contrastive_head projection of the image tower and the
Clinical-Longformer -> Linear(768,512) text tower, BOTH L2-normalised by Merlin.
Streaming: no preprocessed volume is written to disk.

Domain caveat: Merlin is an ABDOMINAL/pelvic CT model; this is chest CT.
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")
from peds_report_text import valid_volumes, texts_for

def log(*a): print(*a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="/temp_work/ch278233/PEDS_BENCH/retrieval/_merlin_shards")
    args = ap.parse_args()

    from merlin.data.monai_transforms import ImageTransforms
    from merlin.models.build import MerlinArchitecture, sanitize_report

    stems = valid_volumes()                    # order of peds23_vollist_valid.txt
    texts = texts_for(stems)                   # canonical byte-identical report strings
    vmap = {}
    with open("/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            vmap[r["volume_name"]] = r["nii_path"]
    gidx = list(range(len(stems)))
    if args.limit: gidx = gidx[:args.limit]
    gidx = gidx[args.shard::args.nshards]
    log(f"[shard {args.shard}/{args.nshards}] n={len(gidx)} of {len(stems)}")

    CKPT = ("/temp_work/ch278233/PEDS_BENCH/weights/merlin/"
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt")
    model = MerlinArchitecture()               # default = full CLIP model (image+text towers)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    msd = model.state_dict()
    matched = sum(1 for k in sd if k in msd and msd[k].shape == sd[k].shape)
    model.load_state_dict(sd, strict=True)
    log(f"[ckpt] ckpt_tensors={len(sd)} model_tensors={len(msd)} matched={matched} strict_load=OK "
        f"logit_scale={float(model.logit_scale):.4f}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    tok = model.encode_text.tokenizer
    cfg = model.encode_text.text_encoder.config
    ctx = {"merlin_max_length": 1024, "tokenizer_model_max_length": int(tok.model_max_length),
           "text_encoder_max_position_embeddings": int(getattr(cfg, "max_position_embeddings", -1))}
    log(f"[text] {ctx}")

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(gidx)
        def __getitem__(self, i):
            g = gidx[i]; name = stems[g] + ".nii.gz"
            try:
                img = ImageTransforms({"image": vmap[name]})["image"]
                if hasattr(img, "as_tensor"): img = img.as_tensor()
                return g, img.float(), True
            except Exception as e:
                sys.stderr.write(f"FAIL {name}: {type(e).__name__}: {e}\n")
                return g, torch.zeros(1, 224, 224, 160), False

    dl = torch.utils.data.DataLoader(DS(), batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, collate_fn=lambda b: b)
    D = 512
    img_lat = np.zeros((len(gidx), D), np.float32); txt_lat = np.zeros((len(gidx), D), np.float32)
    ntok = np.zeros(len(gidx), np.int32); ok = np.zeros(len(gidx), bool)
    pos = {g: i for i, g in enumerate(gidx)}
    failures = []; t0 = time.time(); done = 0
    with torch.no_grad():
        for batch in dl:
            good = [(g, im) for g, im, f in batch if f]
            failures += [stems[g] for g, _, f in batch if not f]
            if good:
                gs = [g for g, _ in good]
                x = torch.stack([im for _, im in good]).to(dev)
                tb = [texts[g] for g in gs]
                # Merlin's own contrastive forward: both towers, both L2-normalised inside
                img_f, ehr_f, txt_f = model(x, tb)
                assert img_f.shape == (len(gs), D) and txt_f.shape == (len(gs), D), (img_f.shape, txt_f.shape)
                for j, g in enumerate(gs):
                    img_lat[pos[g]] = img_f[j].float().cpu().numpy()
                    txt_lat[pos[g]] = txt_f[j].float().cpu().numpy()
                    # token count with Merlin's own sanitisation, NO truncation (for truncation stats)
                    ntok[pos[g]] = len(tok(sanitize_report(texts[g]), truncation=False)["input_ids"])
                    ok[pos[g]] = True
                del x
            done += len(batch)
            if done % 20 < args.batch_size:
                el = time.time() - t0
                log(f"  {done}/{len(gidx)} {el/max(done,1):.2f}s/vol eta={(len(gidx)-done)*el/max(done,1)/60:.1f}min")
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"shard{args.shard:02d}.npz")
    np.savez_compressed(out, img_lat=img_lat, txt_lat=txt_lat, gidx=np.array(gidx), ntok=ntok, ok=ok,
                        stems=np.array([stems[g] for g in gidx]), failed=np.array(failures, dtype=str),
                        ctx=np.array(json.dumps(ctx)))
    log(f"[done] wrote {out} ok={int(ok.sum())}/{len(gidx)} failed={len(failures)} "
        f"truncated(>1024 tok)={int((ntok[ok] > 1024).sum())}")
    if failures: log("[failed] " + ", ".join(failures))

if __name__ == "__main__":
    main()
