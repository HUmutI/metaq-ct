#!/home/ch278233/micromamba/envs/arcct/bin/python
"""Score cached CT-CLIP image latents against prompt templates (CPU, instant).

Needs a text encoder to embed the prompts, so it loads only CXR-BERT + CT-CLIP's
to_text_latent projection from the checkpoint -- not the 3D vision tower. That is
seconds on CPU rather than a 40-minute GPU pass per template.

Reproduces CT-CLIP's own readout exactly: L2-normalise both latents, dot, multiply
by the learned temperature, softmax over the (positive, negative) pair, take P(pos).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/ch278233/BENCHMARK/CT-CLIP/scripts")

TEMPLATES = {
    # scripts/zero_shot.py:134 -- the published run_zero_shot.py entry point
    "present":   lambda c: (f"{c} is present.", f"{c} is not present."),
    # scripts/CTCLIPTrainer.py:293 -- the in-training readout
    "thereis":   lambda c: (f"There is {c}.", f"There is no {c}."),
    # the template ARC-CT and MPS-CT use, for a like-for-like comparison
    "bare_null": lambda c: (f"{c}.", ""),
    "bare_not":  lambda c: (f"{c}.", f"not {c}."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    from transformers import BertTokenizer, BertModel
    from sklearn.metrics import roc_auc_score

    z = np.load(a.latents, allow_pickle=True)
    lat = torch.from_numpy(z["latents"].astype(np.float32))
    true = z["true"].astype(np.int32)
    classes = [str(c) for c in z["classes"]]
    vols = z["volumes"]
    temp = float(z["temperature"])
    print(f"[score] {a.tag}: n={lat.shape[0]} dim={lat.shape[1]} temp={temp:.4f}", flush=True)

    tok = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    te = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    # Text tower + text projection, exactly as stored in the checkpoint.
    tw = {k[len("text_transformer."):]: v for k, v in sd.items() if k.startswith("text_transformer.")}
    missing = te.load_state_dict(tw, strict=False)
    proj_w = sd["to_text_latent.weight"]
    proj_b = sd.get("to_text_latent.bias")
    print(f"[score] text tower: {len(tw)} tensors, proj {tuple(proj_w.shape)}", flush=True)
    te.eval()

    def embed(strings):
        tk = tok(strings, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        with torch.no_grad():
            cls = te(tk.input_ids, attention_mask=tk.attention_mask).last_hidden_state[:, 0, :]
            out = cls @ proj_w.t()
            if proj_b is not None:
                out = out + proj_b
        return out

    os.makedirs(a.out_dir, exist_ok=True)
    il = torch.nn.functional.normalize(lat, dim=-1)
    results = {}
    for name, fn in TEMPLATES.items():
        pos, neg = zip(*[fn(c) for c in classes])
        tl = torch.nn.functional.normalize(embed(list(pos) + list(neg)), dim=-1)
        C = len(classes)
        sims = torch.stack([il @ tl[:C].t(), il @ tl[C:].t()], dim=2) * temp   # (n,C,2)
        pred = torch.softmax(sims, dim=2)[:, :, 0].numpy().astype(np.float32)
        aucs = [roc_auc_score(true[:, j], pred[:, j]) for j in range(C)
                if len(np.unique(true[:, j])) > 1]
        results[name] = float(np.mean(aucs))
        np.savez_compressed(os.path.join(a.out_dir, f"{a.tag}_{name}_preds.npz"),
                            pred=pred, true=true, classes=np.array(classes), volumes=vols)
        print(f"[score] {name:10s} macroAUC={np.mean(aucs):.4f}  ({len(aucs)}/{C} sinif)", flush=True)

    json.dump(results, open(os.path.join(a.out_dir, f"{a.tag}_template_sweep.json"), "w"), indent=2)
    best = max(results, key=results.get)
    print(f"\n[score] en iyi sablon: {best} = {results[best]:.4f} | yayinlanan giris noktasi "
          f"(run_zero_shot.py) = 'present' = {results['present']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
