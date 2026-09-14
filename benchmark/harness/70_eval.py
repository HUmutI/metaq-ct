#!/usr/bin/env python3
"""Zero-shot / classifier AUROC for MPS-CT and GreenRFM on our cohorts.

Neither repo's own evaluation runs as shipped, so the scoring is reimplemented
here against each model's *training* contract rather than its eval script:

MPS-CT, two independent defects in run_zero_shot.py / zero_shot.py:
  1. run_zero_shot.py sets image_encoder.avgpool = nn.Identity(), which makes
     r3d_18 emit 512*6*12*12 = 442,368 features for a 96x192x192 volume, and
     to_visual_latent is Linear(512 -> 768).  Measured: "mat1 and mat2 shapes
     cannot be multiplied (1x442368 and 512x768)".  train_clip.py keeps avgpool
     and yields (B, 512), which is what dim_image=512 means.  We follow
     train_clip.py, because that is what produced the checkpoint.
  2. zero_shot.py calls model(text_tokens, volume, device=...) without
     return_latents and softmaxes the result, but CTCLIP.forward has no return
     statement on that branch -- the body ends on a dangling comment at
     ct_clip.py:714 -- so it returns None.
  We therefore call the two towers directly, reproducing exactly what
  CTCLIP.forward does on the return_latents path, and score with the same
  similarity the loss uses: a raw dot product of unnormalised latents
  (CTCLIPTrainer.compute_loss:337; note l2norm is commented out at ct_clip.py:694).
  Calling the towers separately also avoids feeding forward() a dummy 510 MB
  volume just to encode 36 prompts, and avoids re-running BERT on every batch.
  With text_has_cls_token / visual_has_cls_token / text_causal_mask all False by
  default (ct_clip.py:418-430), forward reduces to precisely these two paths.

GreenRFM, two issues:
  1. inference/zero_shot.py swallows every failure as `except: auc = 0.5`, which
     invents a score for a class that has no positives.  We skip such classes and
     report how many were evaluable, matching the convention already used for the
     ARC-CT age strata.
  2. CTRATEDataset's section != "train" branch (data/ct_rate.py:36) is
     CenterSpatialCrop((192,192,96)) with NO pad, so any volume smaller than the
     crop comes out smaller and the batch cannot stack -- measured: "stack expects
     each tensor to be equal size, but got [1, 96, 192, 192] at entry 0 and
     [1, 71, 154, 154] at entry 2".  That branch cannot have been run.  We override
     it with the deterministic form of the pipeline the model was actually trained
     on -- ResizeWithPadOrCrop((240,240,120)) then CenterSpatialCrop((192,192,96))
     -- which always yields 192x192x96 and keeps training's padding/scale
     behaviour.  This mirrors what MPS-CT does with its own _center_crop_pad.

Scoring modes:
  zeroshot   pos/neg (or pos/null) prompt pair, softmax over the pair, P(positive).
             The only mode MPS-CT supports -- it has no classification head.
  classifier GreenRFM only: the jointly-trained head's per-class logits. This is
             the readout comparable to ARC-CT, which is a supervised classifier.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

PROMPTS = {
    # MPS-CT's shipped choice: positive vs the empty string. The `not {p}`
    # variant is present but commented out at zero_shot.py:335.
    "null": lambda p: (f"{p}.", ""),
    # GreenRFM's default p1, and its p3/p4 variants.
    "p1": lambda p: (f"{p}.", f"not {p}."),
    "p3": lambda p: (f"There is {p}.", f"There is no {p}."),
    "p4": lambda p: (f"{p} is present.", f"{p} is not present."),
}


def auroc_table(labels, scores, classes):
    """Per-class AUROC, skipping classes that cannot have one.

    A class with no positives or no negatives has no defined AUROC. Upstream
    GreenRFM returns 0.5 for these, which silently drags a macro toward chance;
    we exclude them and say how many contributed.
    """
    from sklearn.metrics import roc_auc_score
    rows, used = [], []
    for i, c in enumerate(classes):
        y = labels[:, i]
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if npos == 0 or nneg == 0:
            rows.append({"class": c, "auc": None, "n_pos": npos, "skipped": "degenerate"})
            continue
        a = float(roc_auc_score(y, scores[:, i]))
        rows.append({"class": c, "auc": a, "n_pos": npos})
        used.append(a)
    return rows, (float(np.mean(used)) if used else None), len(used)


def build_mpsct(ckpt, device):
    import torch, torch.nn as nn
    from transformers import BertTokenizer, BertModel
    import torchvision.models.video as models
    tok = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    te = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    # NOTE: run_zero_shot.py also calls resize_token_embeddings here; train_clip.py
    # does not, so the checkpoint has the unresized embedding. Resizing would
    # change the matrix the weights were trained for.
    ie = models.r3d_18(pretrained=False)
    ie.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
    ie.fc = nn.Identity()          # avgpool deliberately kept -- see module docstring
    from ct_clip.ct_clip import CTCLIP
    clip = CTCLIP(image_encoder=ie, text_encoder=te, dim_image=512, dim_text=768,
                  dim_latent=768, extra_latent_projection=False, use_mlm=False,
                  downsample_image_embeds=False, use_all_token_embeds=False)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    own = clip.state_dict()
    shared = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    print(f"[eval] mpsct ckpt {os.path.basename(ckpt)}: {len(shared)}/{len(own)} tensors match", flush=True)
    if len(shared) < 0.9 * len(own):
        sys.exit("[eval] FATAL: checkpoint matches <90% of the model -- wrong architecture")
    clip.load_state_dict(sd, strict=False)
    return clip.to(device).eval(), tok


def build_greenrfm(ckpt, num_classes, device):
    import torch
    from transformers import BertTokenizer
    sys.path.insert(0, "/home/ch278233/BENCHMARK/GreenRFM")
    from models.clip import CTCLIP
    bert = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tok = BertTokenizer.from_pretrained(bert, do_lower_case=True)
    model = CTCLIP(num_classes=num_classes, text_encoder_name=bert,
                   dim_image=512, dim_text=768, dim_latent=768)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model", ck)
    own = model.state_dict()
    shared = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    print(f"[eval] greenrfm ckpt {os.path.basename(ckpt)}: {len(shared)}/{len(own)} tensors match", flush=True)
    if len(shared) < 0.9 * len(own):
        sys.exit("[eval] FATAL: checkpoint matches <90% of the model")
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mpsct", "greenrfm"])
    ap.add_argument("--mode", default="zeroshot", choices=["zeroshot", "classifier"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--prompt", default=None, help="null|p1|p3|p4; default: null for mpsct, p1 for greenrfm")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-retrieval", default="",
                    help="also write img/txt latents for report<->CT retrieval to this npz "
                         "(reports encoded with the canonical pediatric string at the model's own 128-token length)")
    a = ap.parse_args()

    import torch, pandas as pd
    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.mode == "classifier" and a.model != "greenrfm":
        sys.exit("[eval] FATAL: only GreenRFM has a classification head; MPS-CT is contrastive only")

    classes = [c for c in pd.read_csv(a.labels, nrows=1).columns if c != "VolumeName"]
    prompt_id = a.prompt or ("null" if a.model == "mpsct" else "p1")
    print(f"[eval] {a.model}/{a.mode} classes={len(classes)} prompt={prompt_id} dev={device}", flush=True)

    # Each repo's own dataset, so preprocessing matches how the model was trained.
    if a.model == "mpsct":
        sys.path.insert(0, "/home/ch278233/BENCHMARK/MPS-CT")
        from ct_report_inference_dataset import CTReportsInferenceDataset
        ds = CTReportsInferenceDataset(a.tree, a.reports, a.labels)
        collate = None
    else:
        sys.path.insert(0, "/home/ch278233/BENCHMARK/GreenRFM")
        from data.ct_rate import CTRATEDataset
        import monai.transforms as mt
        ds = CTRATEDataset(a.tree, a.reports, a.labels, section="valid")
        # See docstring: upstream's valid transform cannot batch. Deterministic
        # equivalent of the train-time spatial pipeline.
        ds.transform = mt.Compose([
            mt.ResizeWithPadOrCrop((240, 240, 120)),
            mt.CenterSpatialCrop((192, 192, 96)),
            mt.ToTensor(),
        ])
        def collate(b):
            return {"image": torch.stack([x["image"] for x in b]),
                    "labels": torch.stack([x["labels"] for x in b]),
                    "accession": [x["accession"] for x in b]}
    if a.limit:
        ds = torch.utils.data.Subset(ds, range(min(a.limit, len(ds))))
    print(f"[eval] scanned {len(ds)} volumes", flush=True)
    if len(ds) == 0:
        sys.exit("[eval] FATAL: zero volumes scanned")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)

    if a.model == "mpsct":
        model, tok = build_mpsct(a.ckpt, device)
    else:
        model, tok = build_greenrfm(a.ckpt, len(classes), device)

    # Text side: one (positive, negative) pair per class, encoded once.
    def encode_text(strings):
        tk = tok(strings, return_tensors="pt", padding="max_length",
                 truncation=True, max_length=128).to(device)
        if a.model == "mpsct":
            enc = model.text_transformer(tk.input_ids, tk.attention_mask)[0]
            return model.to_text_latent(enc[:, 0, :])
        return model(text_input=dict(tk), return_latents=True)["text_latents"]

    def encode_image(vol):
        if a.model == "mpsct":
            return model.to_visual_latent(model.visual_transformer(vol))
        return model(image_input=vol, return_latents=True)["image_latents"]

    text_lat = None
    if a.mode == "zeroshot":
        pos, neg = zip(*[PROMPTS[prompt_id](c) for c in classes])
        with torch.no_grad():
            tl = encode_text(list(pos) + list(neg))
        C = len(classes)
        text_lat = torch.stack([tl[:C], tl[C:]], dim=1)     # [C, 2, D]
        print(f"[eval] text latents {tuple(text_lat.shape)}", flush=True)

    all_scores, all_labels, all_names, all_img = [], [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if a.model == "mpsct":
                vol, _txt, onehot, _name = batch
                vol = vol.to(device)
                y = np.asarray(onehot, dtype=np.float32) if not torch.is_tensor(onehot) else onehot.numpy()
                all_names.extend([str(n) for n in _name])
            else:
                vol = batch["image"].to(device)
                y = batch["labels"].numpy()
                all_names.extend([str(n) for n in batch["accession"]])

            if a.mode == "classifier":
                # GreenRFM's head is trained with binary_cross_entropy_with_logits
                # (training/mr_supervise.py:23), so its raw output is a logit.
                # AUC is rank-based and unaffected, but Prec/Acc/F1 threshold at
                # 0.5 and need a probability -- apply the head's own inverse link.
                s = torch.sigmoid(
                    model(image_input=vol, return_latents=True)["image_cls_logits"].float()
                ).cpu().numpy()
            else:
                img_lat = encode_image(vol)
                if a.dump_retrieval:
                    all_img.append(img_lat.float().cpu().numpy())
                sims = torch.einsum("bd,ctd->bct", img_lat.float(), text_lat.float())
                s = torch.softmax(sims, dim=2)[:, :, 0].cpu().numpy()

            all_scores.append(s)
            all_labels.append(y)
            if bi % 25 == 0:
                print(f"[eval] batch {bi}/{len(dl)}", flush=True)

    scores = np.concatenate(all_scores, 0)
    labels = np.concatenate(all_labels, 0)

    if a.dump_retrieval:
        # Reports go through the SAME encode_text() the prompts use, i.e. this model's
        # own tokenizer at max_length=128 -- the length both repos train with.
        sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")
        from peds_report_text import texts_for, write_retrieval_npz, stem
        stems = [stem(n) for n in all_names]
        texts = texts_for(stems)
        tl = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                tl.append(encode_text(texts[i:i+32]).float().cpu().numpy())
        write_retrieval_npz(a.dump_retrieval, np.concatenate(all_img, 0), np.concatenate(tl, 0), stems,
                            meta={"model": a.model, "ckpt": os.path.basename(a.ckpt), "text_ctx": 128})
        print(f"[eval] retrieval latents -> {a.dump_retrieval}", flush=True)
    rows, macro, n_used = auroc_table(labels, scores, classes)

    res = {"model": a.model, "mode": a.mode, "prompt": prompt_id if a.mode == "zeroshot" else None,
           "ckpt": a.ckpt, "tree": a.tree, "labels_csv": a.labels,
           "n_volumes": int(labels.shape[0]), "n_classes": len(classes),
           "n_classes_evaluable": n_used, "macro_auroc": macro, "per_class": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    # Save raw predictions so the pediatric benchmark table can be scored with
    # ARC-CT's own compute_metric_bundle -- identical metric code for every row is
    # the only way AUC/Prec/Acc/F1 are comparable across models.
    np.savez_compressed(a.out.replace(".json", "_preds.npz"),
                        pred=scores.astype(np.float32),
                        true=labels.astype(np.int32),
                        classes=np.array(classes),
                        volumes=np.array(all_names))
    print(f"[eval] predictions -> {a.out.replace('.json', '_preds.npz')}", flush=True)
    print(f"[eval] MACRO AUROC = {macro if macro is None else round(macro,4)} "
          f"over {n_used}/{len(classes)} classes, {labels.shape[0]} volumes -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
