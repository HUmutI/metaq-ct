#!/usr/bin/env python3
"""Drive MPS-CT training on our cohorts without editing the upstream clone.

Mirrors BENCHMARK/MPS-CT/train_clip.py exactly -- same encoders, same CTCLIP
construction, same trainer -- but takes every path from the environment so the
same file serves the CT-RATE pretrain and the later pediatric fine-tune, and so
the pristine clone stays untouched for provenance.

Two upstream behaviours must be handled or the batch job dies silently:

  CTCLIPTrainer.py:293 calls yes_or_no() -> input() when results_folder is
  non-empty.  Under sbatch there is no stdin, so that raises EOFError after the
  model and both datasets are already built.  We refuse to start unless the
  results folder is empty, which short-circuits the `and` before the prompt.
  Every GPU partition here is PreemptMode=REQUEUE at the same PriorityTier, and
  two of this account's jobs were preempted in the last week, so a restart is a
  real overnight event -- not an edge case.  The sbatch therefore gives each
  attempt its own results dir and warm-starts it from the newest checkpoint of
  the previous attempts, which keeps that guard intact instead of defeating it.

  CTCLIPTrainer.py:211 calls wandb.init unconditionally.  WANDB_MODE=disabled
  is set by the sbatch wrapper; we assert it here so a missing export cannot
  block on a login prompt on a compute node.
"""
from __future__ import annotations
import os, sys

def env(k, default=None, required=False):
    v = os.environ.get(k, default)
    if required and not v:
        sys.exit(f"[mpsct] FATAL: {k} is unset")
    return v

def main() -> int:
    if os.environ.get("WANDB_MODE") not in ("disabled", "offline"):
        sys.exit("[mpsct] FATAL: set WANDB_MODE=disabled (upstream calls wandb.init unconditionally)")

    data_train    = env("BM_DATA_TRAIN", required=True)
    data_valid    = env("BM_DATA_VALID", required=True)
    reports_train = env("BM_REPORTS_TRAIN", required=True)
    reports_valid = env("BM_REPORTS_VALID", required=True)
    labels        = env("BM_LABELS", required=True)
    results       = env("BM_RESULTS", required=True)

    steps   = int(env("BM_STEPS", "30001"))
    batch   = int(env("BM_BATCH", "10"))
    workers = int(env("BM_WORKERS", "4"))
    lr      = float(env("BM_LR", "1e-5"))
    save_every = int(env("BM_SAVE_EVERY", "1000"))
    pretrained = env("BM_PRETRAINED", "") or None

    for p in (data_train, data_valid, reports_train, reports_valid, labels):
        if not os.path.exists(p):
            sys.exit(f"[mpsct] FATAL: missing input {p}")

    os.makedirs(results, exist_ok=True)
    leftover = os.listdir(results)
    if leftover:
        sys.exit(f"[mpsct] FATAL: results folder {results} is not empty ({len(leftover)} entries). "
                 "Upstream would prompt on stdin and hang. Move it aside or pick a new dir.")

    print(f"[mpsct] paths OK, results={results}; importing torch ...", flush=True)
    import torch, torch.nn as nn
    from transformers import BertTokenizer, BertModel
    import torchvision.models.video as models

    print(f"[mpsct] imports done, torch {torch.__version__}; probing CUDA ...", flush=True)
    _avail = torch.cuda.is_available()
    print(f"[mpsct] cuda={_avail} dev={torch.cuda.get_device_name(0) if _avail else 'CPU'}", flush=True)

    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    # train_clip.py replaces the stem for single-channel CT and drops the fc.
    # It deliberately KEEPS avgpool, which is what makes dim_image=512 correct.
    image_encoder = models.r3d_18(pretrained=True)
    image_encoder.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
    image_encoder.fc = nn.Identity()

    from ct_clip.ct_clip import CTCLIP
    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=512,
        dim_text=768,
        dim_latent=768,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )

    # Upstream cannot load its own warm start: CTCLIPTrainer.py:225 calls
    # self.CTClip.load(path, map_location=...) but CTCLIP.load (ct_clip.py:567)
    # takes only `path`, so passing pretrained_path raises TypeError.  That code
    # path has evidently never been run -- their workflow trains from scratch.
    # We therefore load the weights ourselves and hand the trainer None.
    #
    # CTCLIP.load also uses strict=False, which silently tolerates a total key
    # mismatch, so we count the overlap and refuse a warm start that matched
    # nothing rather than quietly fine-tuning from random init.
    if pretrained:
        import collections
        sd = torch.load(pretrained, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        own = clip.state_dict()
        shared = [k for k in sd if k in own and own[k].shape == sd[k].shape]
        print(f"[mpsct] warm start {pretrained}: {len(shared)}/{len(sd)} tensors match "
              f"(model has {len(own)})", flush=True)
        if not shared:
            sys.exit("[mpsct] FATAL: warm start matched ZERO tensors -- refusing to "
                     "fine-tune from random init")
        missing, unexpected = clip.load_state_dict(sd, strict=False)
        print(f"[mpsct] loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    from CTCLIPTrainer import CTClipTrainer
    trainer = CTClipTrainer(
        clip,
        reports_file_train=reports_train,
        reports_file_valid=reports_valid,
        data_train=data_train,
        data_valid=data_valid,
        labels=labels,
        tokenizer=tokenizer,
        batch_size=batch,
        results_folder=results,
        num_train_steps=steps,
        num_workers=workers,
        lr=lr,
        max_grad_norm=0.5,
        save_results_every=save_every,
        save_model_every=save_every,
        pretrained_path=None,   # already loaded above; see note
    )
    print(f"[mpsct] train samples={len(trainer.ds)} valid samples={len(trainer.valid_ds)}", flush=True)
    if len(trainer.ds) == 0:
        sys.exit("[mpsct] FATAL: dataset scanned to ZERO samples -- check tree depth and VolumeName keys")
    trainer.train()
    return 0

if __name__ == "__main__":
    sys.exit(main())
