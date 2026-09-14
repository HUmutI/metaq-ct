#!/usr/bin/env python
"""Merge Merlin contrastive shards -> /temp_work/ch278233/PEDS_BENCH/retrieval/merlin_valid.npz"""
import glob, json, sys, numpy as np
sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")
from peds_report_text import valid_volumes, texts_for, write_retrieval_npz

SH = "/temp_work/ch278233/PEDS_BENCH/retrieval/_merlin_shards"
OUT = "/temp_work/ch278233/PEDS_BENCH/retrieval/merlin_valid.npz"
stems = valid_volumes(); N = len(stems)
img = np.full((N, 512), np.nan, np.float32); txt = np.full((N, 512), np.nan, np.float32)
ntok = np.zeros(N, np.int32); seen = np.zeros(N, bool); failed = []; ctx = None
SHARD_FILES = sorted(glob.glob(f"{SH}/shard*.npz")) + sorted(glob.glob(f"{SH}_rescue/shard*.npz"))
for f in SHARD_FILES:
    d = np.load(f, allow_pickle=True)
    for i, g in enumerate(d["gidx"]):
        assert d["stems"][i] == stems[g], (f, i, d["stems"][i], stems[g])
        if d["ok"][i]:
            assert not seen[g], f"duplicate {stems[g]}"
            img[g] = d["img_lat"][i]; txt[g] = d["txt_lat"][i]; ntok[g] = d["ntok"][i]; seen[g] = True
    failed += list(d["failed"]); ctx = json.loads(str(d["ctx"]))
print(f"shards={len(SHARD_FILES)} covered={int(seen.sum())}/{N} failed={failed}")
assert seen.all(), f"missing {[stems[i] for i in np.where(~seen)[0]][:10]}"
assert not np.isnan(img).any() and not np.isnan(txt).any()
ni, nt = np.linalg.norm(img, axis=1), np.linalg.norm(txt, axis=1)
trunc = int((ntok > ctx["merlin_max_length"]).sum())
meta = {"model": "Merlin (StanfordMIMI) i3-ResNet152 contrastive_head (512-d) + Clinical-Longformer->Linear(768,512)",
        "ckpt": "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt", "ckpt_tensors_matched": "1208/1208",
        "D": 512, "l2_normalised_by": "MerlinArchitecture.forward", "text_sanitised_by": "merlin.models.build.sanitize_report (lower+wordpunct)",
        "context_tokens": ctx["merlin_max_length"], "tokenizer_model_max_length": ctx["tokenizer_model_max_length"],
        "n_truncated_reports": trunc, "ntok_max": int(ntok.max()), "ntok_median": float(np.median(ntok)),
        "domain_caveat": "Merlin is an ABDOMINAL/pelvic CT model applied to PEDIATRIC CHEST CT (out-of-domain)."}
write_retrieval_npz(OUT, img, txt, stems, meta=meta)
d = np.load(OUT, allow_pickle=True)
print("wrote", OUT); print({k: (d[k].shape, str(d[k].dtype)) for k in d.files})
print(f"D=512 mean|img|={ni.mean():.6f} mean|txt|={nt.mean():.6f} min/max img {ni.min():.6f}/{ni.max():.6f} txt {nt.min():.6f}/{nt.max():.6f}")
print(f"truncated(>{ctx['merlin_max_length']} tok)={trunc}/{N}  ntok median={np.median(ntok):.0f} max={ntok.max()}")
print("accessions == vollist order:", list(d["accessions"]) == stems, "| row check:", np.allclose(d["img_lat"][stems.index('ped_00010_1')], img[stems.index('ped_00010_1')]))
print("text_ok all:", bool(d["text_ok"].all()), "labels", d["labels"].shape, d["labels"].dtype, "classes", len(d["classes"]))
cos = (d["img_lat"] * d["txt_lat"]).sum(1); print(f"paired cos mean={cos.mean():.4f}")
tx = texts_for(stems)
for s in ["ped_00002_1", "ped_00010_1"]: print(f"  {s}: {tx[stems.index(s)][:60]!r}")
