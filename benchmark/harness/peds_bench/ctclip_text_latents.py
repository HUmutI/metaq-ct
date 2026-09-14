#!/usr/bin/env python3
"""CT-CLIP / VocabFine report latents in the same space as the cached image latents.
Text tower = CXR-BERT weights + to_text_latent from the checkpoint (as score_latents.py),
CLS -> projection -> L2-normalise (ct_clip.py normalises both towers). Context length
512 = CT-CLIP's own (zero_shot.py / CTCLIPTrainer.py)."""
import argparse, sys, os, numpy as np, torch
sys.path.insert(0, "/home/ch278233/BENCHMARK/CT-CLIP/scripts"); sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")
from peds_report_text import texts_for, write_retrieval_npz
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--latents", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import BertTokenizer, BertModel
dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(a.latents, allow_pickle=True); vols = [str(v) for v in z["volumes"]]; img = z["latents"].astype(np.float32)
texts = texts_for(vols)
tok = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
te = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
sd = torch.load(a.ckpt, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) else sd
te.load_state_dict({k[len("text_transformer."):]: v for k, v in sd.items() if k.startswith("text_transformer.")}, strict=False)
W = sd["to_text_latent.weight"].to(dev); B = sd.get("to_text_latent.bias"); B = B.to(dev) if B is not None else None
te = te.to(dev).eval(); out = []; ntrunc = 0
with torch.no_grad():
    for i in range(0, len(texts), 32):
        tk = tok(texts[i:i+32], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        ntrunc += int((tk.attention_mask.sum(1) == 512).sum())
        tk = tk.to(dev); cls = te(tk.input_ids, attention_mask=tk.attention_mask).last_hidden_state[:, 0, :]
        lat = cls @ W.t() + (B if B is not None else 0)
        out.append(torch.nn.functional.normalize(lat, dim=-1).float().cpu().numpy())
txt = np.concatenate(out)
write_retrieval_npz(a.out, img, txt, vols, meta={"ckpt": os.path.basename(a.ckpt), "text_ctx": 512, "n_truncated": ntrunc})
print(f"[ctclip-txt] {os.path.basename(a.ckpt)}: img{img.shape} txt{txt.shape} truncated@512={ntrunc}/{len(texts)} -> {a.out}")
