#!/usr/bin/env python3
"""Fine-tune official CT-CLIP on the pediatric cohort, then leave it ready for
the zero-shot prompt readout that CT-CLIP's paper reports.

Protocol mirrors what was done for GreenRFM / MPS-CT: CT-RATE pretrain, then
pediatric fine-tune.  The difference is that CT-CLIP *publishes* its CT-RATE
weights, so we start from the official CT-CLIP_v2.pt rather than a retrain.
That makes this baseline stronger than a from-scratch one, which is the
conservative direction for our own claim.

Nothing about the model is modified: CTViT + CXR-BERT are constructed exactly as
scripts/run_train.py does, and CTClipTrainer runs unmodified.  Only the two
dataset classes are swapped, because our cohort is a flat vollist of cached
tensors rather than CT-RATE's nested NIfTI tree.
"""
from __future__ import annotations
import argparse, os, sys, torch

CTCLIP_SCRIPTS = "/home/ch278233/BENCHMARK/CT-CLIP/scripts"
sys.path.insert(0, CTCLIP_SCRIPTS)
sys.path.insert(0, "/home/ch278233/BENCHMARK/harness/peds_bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="/temp_work/ch278233/BENCHMARK_WEIGHTS/CT-CLIP/CT-CLIP_v2.pt")
    ap.add_argument("--train-vollist", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_train.txt")
    ap.add_argument("--valid-vollist", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt")
    ap.add_argument("--reports", default="/temp_work/ch278233/COMBINED_reports.csv")
    ap.add_argument("--labels", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv")
    ap.add_argument("--out", default="/temp_work/ch278233/PEDS_BENCH/runs/ctclip_peds")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.25e-6)   # CT-CLIP's own default
    ap.add_argument("--workers", type=int, default=6)
    # Upstream defaults are save_model_every=1 / save_results_every=1, i.e. a
    # multi-GB checkpoint AND a full validation pass on every single step. Those
    # are harness cadences, not model hyperparameters; leaving them at 1 would
    # make the run impossible rather than faithful.
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=2000)
    a = ap.parse_args()

    from transformer_maskgit import CTViT
    from transformers import BertTokenizer, BertModel
    from ct_clip import CTCLIP
    # Minimal-diff copy of CT-CLIP's trainer: the ONLY change is that the
    # in-training diagnostic readout's hardcoded 18-class CT-RATE pathology list
    # is overridable (upstream crashes at step 0 on a 23-class cohort). The
    # contrastive objective, optimizer and weights are byte-identical upstream.
    import CTCLIPTrainer_peds as CTCLIPTrainer
    from peds_ctclip_data import PedsCTReportDataset, PedsCTReportDatasetInfer

    # Swap ONLY the data layer; the trainer itself is untouched.
    CTCLIPTrainer.CTReportDataset = PedsCTReportDataset
    CTCLIPTrainer.CTReportDatasetinfer = PedsCTReportDatasetInfer

    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
                          temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
                          dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder,
                  dim_text=768, dim_image=294912, dim_latent=512,
                  extra_latent_projection=False, use_mlm=False,
                  downsample_image_embeds=False, use_all_token_embeds=False)

    sd = torch.load(a.init, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    own = clip.state_dict()
    shared = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    print(f"[ctclip] init {os.path.basename(a.init)}: {len(shared)}/{len(own)} tensors match", flush=True)
    if len(shared) < 0.9 * len(own):
        sys.exit(f"[ctclip] FATAL: official weights match only {len(shared)}/{len(own)} tensors -- "
                 "architecture mismatch, refusing to train a differently-shaped model")
    clip.load_state_dict(sd, strict=False)

    os.makedirs(a.out, exist_ok=True)
    trainer = CTCLIPTrainer.CTClipTrainer(
        clip,
        reports_file_train=a.reports, reports_file_valid=a.reports,
        data_train=a.train_vollist, data_valid=a.valid_vollist,
        train_meta_file=None, valid_meta_file=None,
        labels=a.labels,
        batch_size=a.batch, results_folder=a.out,
        num_train_steps=a.steps, num_workers=a.workers,
        lr=a.lr, tokenizer=tokenizer,
        save_model_every=a.save_every, save_results_every=a.val_every,
    )
    import pandas as pd
    classes = [c for c in pd.read_csv(a.labels, nrows=1).columns if c != "VolumeName"]
    trainer.pathologies_override = classes
    print(f"[ctclip] in-training diagnostic readout over {len(classes)} pediatric classes",
          flush=True)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
