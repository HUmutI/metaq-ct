"""Final RAC-CLIP stage-2 training for the TIA submission.

Recipe implemented here:
  * 3-channel CT input (lung / soft / bone windows) from RACDatasetV4
  * R3D-18 Kinetics image encoder, optionally initialized from stage-1
  * CXR-BERT with LoRA adapters on last 4 query/value projections
  * Three losses: soft CLIP, global prompt BCE, per-organ report alignment
  * Global mask-free validation AUC is the checkpoint-selection metric

Masks are used only for the anatomy-aware training loss. The primary validation
metric remains mask-free global AUC to avoid selecting on oracle-routed masks.
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", f"{HERE}/.cache/torch")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))

from arcct.dataset import (  # noqa: E402
    ORGAN_TO_PATHIDX,
    PATHOLOGIES,
    PATHOLOGY_FINE_ORGANS,
    N_PATH,
    RACDatasetV4,
    rac_collate,
    region_text_from_cache,
)
from arcct.lora import apply_lora_to_bert, is_lora_parameter  # noqa: E402
from arcct.image_encoder import RACImageEncoder, load_from_stage1  # noqa: E402
from arcct.ct_clip import CTCLIP  # noqa: E402
from arcct.qformer import QFormer
from arcct.context import NO_INDICATION
from arcct.context_qformer import ContextQFormer
from arcct.relevance import RelevanceHead
from arcct.slots import CtxConfig, SlotLayout  # noqa: E402
from arcct.anatomy_qformer import AnatomyQFormer, build_role_mask  # noqa: E402


DATA_ROOT = os.environ.get("RAC_DATA_ROOT", "/mnt/amax5_drive/alp_ozaydin_0/data")
# Overridable so a cohort that is not laid out as mps_ct_npz/<split> can be
# pointed at directly. Without this the only way to train on another dataset was
# to fake CT-RATE's directory tree, and the reports path below had the same
# problem - both were derived from RAC_DATA_ROOT with the rest hardcoded.
DATA_TRAIN = os.environ.get("RAC_DATA_TRAIN", f"{DATA_ROOT}/mps_ct_npz/train")
DATA_VALID = os.environ.get("RAC_DATA_VALID", f"{DATA_ROOT}/mps_ct_npz/valid")
MASK_TRAIN = os.environ.get("RAC_MASK_TRAIN", "/mnt/amax2_drive/alp_ozaydin_0/data/fine_train_masks_192")
MASK_VALID = os.environ.get("RAC_MASK_VALID", "/mnt/amax2_drive/alp_ozaydin_0/data/fine_valid_masks_192")
REPORTS_TRAIN = os.environ.get("RAC_REPORTS_TRAIN", f"{DATA_ROOT}/ct_reports/train_reports.csv")
REPORTS_VALID = os.environ.get("RAC_REPORTS_VALID", f"{DATA_ROOT}/ct_reports/valid_reports.csv")
LABELS_TRAIN = os.environ.get("RAC_LABELS_TRAIN", f"{DATA_ROOT}/multi_abnormality_labels/train_predicted_labels.csv")
LABELS_VALID = os.environ.get("RAC_LABELS_VALID", f"{DATA_ROOT}/multi_abnormality_labels/valid_predicted_labels.csv")
REGION_CACHE = os.environ.get("RAC_REGION_CACHE", "")
STAGE1_CKPT = os.environ.get("RAC_STAGE1_CKPT", "")
RESULTS_DIR = os.environ.get("RAC_RESULTS_DIR", f"{HERE}/runs/final_seed{os.environ.get('RAC_SEED', '0')}")

BATCH_SIZE = int(os.environ.get("RAC_BATCH_SIZE", "4"))
ACCUM_STEPS = int(os.environ.get("RAC_ACCUM_STEPS", "8"))
TOTAL_UPDATES = int(os.environ.get("RAC_TOTAL_UPDATES", os.environ.get("RAC_TOTAL_STEPS", "6000")))
WARMUP_UPDATES = int(os.environ.get("RAC_WARMUP_UPDATES", os.environ.get("RAC_WARMUP_STEPS", "500")))
SAVE_EVERY = int(os.environ.get("RAC_SAVE_EVERY", "500"))
VAL_EVERY = int(os.environ.get("RAC_VAL_EVERY", "500"))
PROMPT_CACHE_EVERY = int(os.environ.get("RAC_PROMPT_CACHE_EVERY", "50"))

LOG_EVERY = int(os.environ.get("RAC_LOG_EVERY", "50"))
NUM_WORKERS = int(os.environ.get("RAC_NUM_WORKERS", "6"))
TRAIN_LIMIT = int(os.environ.get("RAC_TRAIN_LIMIT", "0"))
VAL_LIMIT = int(os.environ.get("RAC_VAL_LIMIT", "0"))
FAIL_FAST = os.environ.get("RAC_FAIL_FAST", "1") == "1"

VISION_LR = float(os.environ.get("RAC_LR", "1e-5"))
TEXT_LR = float(os.environ.get("RAC_TEXT_LR", "1e-7"))
WEIGHT_DECAY = float(os.environ.get("RAC_WEIGHT_DECAY", "1e-4"))
TEXT_WEIGHT_DECAY = float(os.environ.get("RAC_TEXT_WEIGHT_DECAY", "1e-3"))
MAX_GRAD_NORM = float(os.environ.get("RAC_MAX_GRAD_NORM", "0.5"))
TEMPERATURE = float(os.environ.get("RAC_TEMPERATURE", "0.07"))

# MD row 4.2: learnable temperature (CLIP-style log_logit_scale).
# When enabled, init to log(1/TEMPERATURE), clamp max so τ stays >= 1/100 = 0.01.
USE_LEARNABLE_TEMP = bool(int(os.environ.get("RAC_USE_LEARNABLE_TEMP", "0")))
LEARNABLE_TEMP_MAX_SCALE = float(os.environ.get("RAC_LEARNABLE_TEMP_MAX_SCALE", "100.0"))
CLIP_WEIGHT = float(os.environ.get("RAC_CLIP_WEIGHT", "1.0"))
CLS_WEIGHT = float(os.environ.get("RAC_CLS_WEIGHT", "0.5"))
ALIGN_WEIGHT = float(os.environ.get("RAC_ALIGN_WEIGHT", "1.0"))
FN_WEIGHT = float(os.environ.get("RAC_FN_WEIGHT", "0.3"))
MAX_TEXT_LEN = int(os.environ.get("RAC_MAX_TEXT_LEN", "256"))
PROMPT_LEN = int(os.environ.get("RAC_PROMPT_LEN", "64"))
USE_LORA = os.environ.get("RAC_USE_LORA", "1") == "1"
LORA_R = int(os.environ.get("RAC_LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("RAC_LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("RAC_LORA_DROPOUT", "0.0"))
USE_AMP = os.environ.get("RAC_AMP", "1") == "1"
KINETICS_PRETRAINED = os.environ.get("RAC_KINETICS_PRETRAINED", "1") == "1"
EARLY_STOP_PATIENCE = int(os.environ.get("RAC_EARLY_STOP_PATIENCE", "4"))
EARLY_STOP_MIN_UPDATES = int(os.environ.get("RAC_EARLY_STOP_MIN_UPDATES", "1000"))

# AnTS-HardNeg controls (env-gated; default off so existing behavior is intact).
ANTS_WEIGHT = float(os.environ.get("RAC_ANTS_WEIGHT", "0.3"))
ANTS_TAU_HN = float(os.environ.get("RAC_ANTS_TAU_HN", "0.05"))
ANTS_LAMBDA_T = float(os.environ.get("RAC_ANTS_LAMBDA_T", "0.5"))
ANTS_LAMBDA_V = float(os.environ.get("RAC_ANTS_LAMBDA_V", "0.5"))
ANTS_COOCCUR_THRESHOLD = float(os.environ.get("RAC_ANTS_COOCCUR_THRESHOLD", "0.4"))
ANTS_TARGET_CLASSES = [
    s.strip() for s in os.environ.get(
        "RAC_ANTS_TARGET_CLASSES",
        "Lung nodule,Pulmonary fibrotic sequela,Mosaic attenuation pattern",
    ).split(",") if s.strip()
]

# L_anat: Anatomy Evidence Loss — per-class inside vs outside-mask grounding
# pressure. BCE on inside-mask score + margin on outside < inside. Cheap
# (no text fwd, 2 extra pools per class). Targets TIA grounding-faithfulness.
ANAT_MARGIN = float(os.environ.get("RAC_ANAT_MARGIN", "0.05"))
ANAT_SCALE = float(os.environ.get("RAC_ANAT_SCALE", "10.0"))

# SigLIP head (env-gated). Adds per-pair sigmoid loss with label-aware
# positives + cooccur-based FN masking. Independent of batch normalization,
# helps at small physical batch where InfoNCE is dilution-starved.
SIGLIP_WEIGHT = float(os.environ.get("RAC_SIGLIP_WEIGHT", "0.3"))
SIGLIP_POS_THRESH = float(os.environ.get("RAC_SIGLIP_POS_THRESH", "0.5"))
SIGLIP_FN_THRESH = float(os.environ.get("RAC_SIGLIP_FN_THRESH", "0.2"))
SIGLIP_COOCCUR_THRESH = float(os.environ.get("RAC_SIGLIP_COOCCUR_THRESH", "0.4"))
SIGLIP_T_INIT = float(os.environ.get("RAC_SIGLIP_T_INIT", "10.0"))
SIGLIP_B_INIT = float(os.environ.get("RAC_SIGLIP_B_INIT", "-10.0"))

# Per-pathology-token contrastive aux loss (AnatomyQFormer only).
# Each pathology query token (slots A..A+P-1 of qf_tokens) gets L2-normalized
# and cosine'd against its class's pos/neg prompt embedding; BCE-style two-way
# softmax with the label as target. Direct supervision on the region-bound
# token, so the mean-pool of 32 tokens stops being the only training signal.
PERTOKEN_NCE_WEIGHT = float(os.environ.get("RAC_PERTOKEN_NCE_WEIGHT", "0.0"))

# Slice top-K BCE aux loss (AnatomyQFormer only). Per-slice anatomy-routed
# QFormer forward, take pathology query tokens, score per slice, top-K mean,
# BCE vs multilabel target. Matches eval-time slice+anatomy routing path so
# the model is trained for the inference policy that wins (+0.022 at eval).
SLICE_TOPK_TRAIN_SLICES = int(os.environ.get("RAC_SLICE_TOPK_TRAIN_SLICES", "4"))

# Attention-supervision aux loss (AnatomyQFormer only). Extra QFormer forward
# WITHOUT hard routing mask, then penalize pathology-query attention mass
# falling OUTSIDE the organ bbox. Only positive samples contribute (label=1).
# Goal: model learns the routing instead of having it enforced.

# Saliency-MIL aux loss (AnatomyQFormer only). Reuses the same maskless
# forward as attn-sup; for each positive pathology, top-K voxels of attention
# should overlap with the organ bbox. Loss = 1 - (in-mask fraction of top-K).
SALIENCY_MIL_K = int(os.environ.get("RAC_SALIENCY_MIL_K", "32"))

# Impression-aware aux contrastive (v14). Soft-CLIP InfoNCE between img_lat
# and the Impressions-only text latent (in addition to the main loss on
# findings+impressions). Targets cases like pulmonary fibrotic sequela where
# the impression phrase ("minimal fibrotic densities") carries the diagnostic
# signal that gets diluted by the longer findings prefix.

# Variant A: voxel-level TopK pooling for focal classes (Lung nodule etc.).
USE_TOPK_POOL = os.environ.get("RAC_USE_TOPK_POOL", "0") == "1"
TOPK_K = int(os.environ.get("RAC_TOPK_K", "8"))
FOCAL_CLASSES = {
    s.strip() for s in os.environ.get(
        "RAC_FOCAL_CLASSES",
        "Lung nodule,Atelectasis,Lung opacity,Consolidation",
    ).split(",") if s.strip()
}

# Variant B: multi-scale (layer3+layer4) feature pyramid.

# Variant C: salient-patch auxiliary loss.

# Path 1/2/3/4: BLIP-2-style Q-Former bridge.
# QFORMER replaces the global to_visual_latent projection with K=32 learnable
# queries that cross-attend to layer4 spatial features. ANATOMY_QFORMER uses
# per-query masked cross-attention bound to TS organ masks. ITM adds a binary
# match head trained with in-batch hard negatives. GENERATIVE adds a frozen
# GPT-2 small decoder fed from the Q-Former tokens for a report LM loss.
USE_QFORMER = os.environ.get("RAC_USE_QFORMER", "0") == "1"
USE_ANATOMY_QFORMER = os.environ.get("RAC_USE_ANATOMY_QFORMER", "0") == "1"
QFORMER_NUM_QUERIES = int(os.environ.get("RAC_QFORMER_QUERIES", "32"))
QFORMER_DEPTH = int(os.environ.get("RAC_QFORMER_DEPTH", "4"))
QFORMER_HEADS = int(os.environ.get("RAC_QFORMER_HEADS", "8"))
QFORMER_POOL = os.environ.get("RAC_QFORMER_POOL", "mean")
QFORMER_DIM = 768  # CXR-BERT latent dim
LM_MAX_LEN = int(os.environ.get("RAC_LM_MAX_LEN", "128"))

# Phase 1: the indication-conditioned bank. Every flag lives in CtxConfig and
# defaults OFF, so an unset environment reproduces today's run exactly. The
# weights below are the only two the training loop reads directly.
CTX_CFG = CtxConfig.from_env()
USE_CONTEXT_QFORMER = CTX_CFG.enabled
CTX_PTOK_WEIGHT = CTX_CFG.ptok_weight
CF_WEIGHT = float(os.environ.get("RAC_CF_WEIGHT", "0.5"))
CF_TAU = float(os.environ.get("RAC_CF_TAU", "0.5"))
CF_MAX_ROWS = int(os.environ.get("RAC_CF_MAX_ROWS", "6"))

ORGAN_LABELS = sorted(ORGAN_TO_PATHIDX)

# Indices of the focal classes within PATHOLOGIES. pool_organ_masks_topk has
# been computed on every masked step and thrown away - organ_lat_focal was
# assigned and never read - so the focal-class configuration has been inert
# while still paying for the extra pooling and projection.
#
# Mean-pooling is the wrong operator for these: a 5 mm nodule averaged over a
# whole lobe's tokens is diluted to nothing, which is visible in the results -
# Lung nodule 0.671 and Bone lesion 0.613 are the weakest classes, and the
# adult model only reaches 0.759 on Lung nodule either. Top-K keeps the k
# strongest tokens instead.
#
# Off by default: it changes the align target for the focal classes, so it must
# not silently move a number that is being compared against.
FOCAL_PATHIDX = frozenset(
    i for i, n in enumerate(PATHOLOGIES) if n in FOCAL_CLASSES
)
USE_FOCAL_ALIGN = os.environ.get("RAC_FOCAL_ALIGN", "0") == "1"

_seed_env = os.environ.get("RAC_SEED", "").strip()
if _seed_env:
    SEED = int(_seed_env)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _GEN = torch.Generator()
    _GEN.manual_seed(SEED)

    def _worker(wid):
        ws = (SEED + wid) & 0xFFFFFFFF
        np.random.seed(ws)
        random.seed(ws)
else:
    _GEN = None
    _worker = None


def _loader_kwargs():
    return {"generator": _GEN, "worker_init_fn": _worker} if _GEN else {}


def _amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return torch.bfloat16 if bf16_ok else torch.float16


def amp_context(device):
    if USE_AMP and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=_amp_dtype())
    return nullcontext()


def build_model():
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    if USE_LORA:
        n_wrapped = apply_lora_to_bert(text_encoder, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        print(f"[RAC] Applied LoRA to {n_wrapped} CXR-BERT projections (r={LORA_R}, alpha={LORA_ALPHA})")

    image_encoder = RACImageEncoder(pretrained_kinetics=KINETICS_PRETRAINED)
    if STAGE1_CKPT and os.path.isfile(STAGE1_CKPT):
        load_from_stage1(image_encoder, STAGE1_CKPT)

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=512,
        dim_text=768,
        dim_latent=768,
        extra_latent_projection=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )
    return clip, tokenizer


def configure_trainable(clip):
    for p in clip.parameters():
        p.requires_grad = False
    for p in clip.visual_transformer.parameters():
        p.requires_grad = True
    for p in clip.to_visual_latent.parameters():
        p.requires_grad = True
    for p in clip.to_text_latent.parameters():
        p.requires_grad = True
    for p in clip.text_transformer.parameters():
        if is_lora_parameter(p):
            p.requires_grad = True

    text_params = []
    vision_params = []
    text_param_ids = set()
    for p in clip.to_text_latent.parameters():
        if p.requires_grad:
            text_params.append(p)
            text_param_ids.add(id(p))
    for p in clip.text_transformer.parameters():
        if p.requires_grad:
            text_params.append(p)
            text_param_ids.add(id(p))
    for p in clip.parameters():
        if p.requires_grad and id(p) not in text_param_ids:
            vision_params.append(p)

    print(f"[RAC] Trainable vision/proj params: {sum(p.numel() for p in vision_params):,}")
    print(f"[RAC] Trainable text LoRA/proj params: {sum(p.numel() for p in text_params):,}")
    groups = [{"params": vision_params, "lr": VISION_LR, "weight_decay": WEIGHT_DECAY}]
    if text_params:
        groups.append({"params": text_params, "lr": TEXT_LR, "weight_decay": TEXT_WEIGHT_DECAY})
    trainable = vision_params + text_params
    return trainable, groups


@torch.no_grad()
def encode_prompts(clip, tokenizer, device):
    pos = tokenizer([f"{p}." for p in PATHOLOGIES], return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_LEN).to(device)
    neg = tokenizer([f"No {p}." for p in PATHOLOGIES], return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_LEN).to(device)
    out_pos = clip.text_transformer(pos["input_ids"], pos["attention_mask"])
    out_neg = clip.text_transformer(neg["input_ids"], neg["attention_mask"])
    pos_embs = F.normalize(clip.to_text_latent(out_pos[0][:, 0, :]), dim=-1)
    neg_embs = F.normalize(clip.to_text_latent(out_neg[0][:, 0, :]), dim=-1)
    return pos_embs, neg_embs


def encode_texts(clip, tokenizer, texts, device):
    toks = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TEXT_LEN).to(device)
    out = clip.text_transformer(toks["input_ids"], toks["attention_mask"])
    return clip.to_text_latent(out[0][:, 0, :])


def encode_token_pack(clip, token_pack, device):
    """Encode pre-tokenized inputs through the live text path.

    The hard-neg bank stores tokenized strings so the text projection follows
    LoRA updates. We re-run only K_c << B sentences per class per step, so
    cost is negligible.
    """
    input_ids = token_pack["input_ids"].to(device)
    attention_mask = token_pack["attention_mask"].to(device)
    out = clip.text_transformer(input_ids, attention_mask)
    return clip.to_text_latent(out[0][:, 0, :])


def soft_clip_infonce(img_lat, txt_lat, labels, temperature=TEMPERATURE, fn_weight=FN_WEIGHT):
    B = img_lat.size(0)
    img_lat = F.normalize(img_lat, dim=-1)
    txt_lat = F.normalize(txt_lat, dim=-1)
    logits = img_lat @ txt_lat.T / temperature
    # Unlabeled cells (NaN) count as "not asserted positive" here, not as a
    # negative. This is a soft-POSITIVE weight over the CLIP target, so an
    # unknown cell must not manufacture similarity between two studies. It also
    # has to happen inside this function: a single NaN turns the whole Jaccard
    # row NaN, then the loss, then every gradient - silently, with no error.
    lab = torch.nan_to_num(labels.float(), nan=0.0)
    inter = lab @ lab.T
    union = lab.sum(1, keepdim=True) + lab.sum(1, keepdim=True).T - inter
    jaccard = inter / union.clamp(min=1.0)
    eye = torch.eye(B, device=img_lat.device)
    soft = (1.0 - fn_weight) * eye + fn_weight * jaccard
    soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-6)
    loss_i = -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_t = -(soft * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return (loss_i + loss_t) / 2


def prompt_probs(img_lat, pos_embs, neg_embs, temperature=TEMPERATURE):
    img_lat = F.normalize(img_lat, dim=-1)
    sims_pos = img_lat @ pos_embs.T
    sims_neg = img_lat @ neg_embs.T
    stacked = torch.stack([sims_pos, sims_neg], dim=-1) / temperature
    return F.softmax(stacked, dim=-1)[..., 0]


def prompt_cls_loss(img_lat, pos_embs, neg_embs, labels, temperature=TEMPERATURE):
    # BCE on probabilities is explicitly kept in fp32 because PyTorch marks
    # BCELoss unsafe under autocast. The surrounding encoder forward can still
    # use AMP.
    with torch.cuda.amp.autocast(enabled=False):
        temp_in = temperature.float() if torch.is_tensor(temperature) else temperature
        probs = prompt_probs(img_lat.float(), pos_embs.float(), neg_embs.float(), temperature=temp_in).clamp(1e-6, 1 - 1e-6)
        tgt = labels.float().to(img_lat.device)
        # F.binary_cross_entropy REJECTS a NaN target outright, so this cannot
        # be forgotten - but the mean must still be over labeled cells only.
        m = torch.isfinite(tgt)
        per = F.binary_cross_entropy(probs, torch.nan_to_num(tgt, nan=0.0),
                                     reduction="none")
        return (per * m).sum() / m.sum().clamp(min=1)


def per_class_auc(pred, true):
    out = {}
    for j, name in enumerate(PATHOLOGIES):
        yt, ys = true[:, j], pred[:, j]
        keep = np.isfinite(yt)                    # unlabeled rows are not evidence
        yt, ys = yt[keep], ys[keep]
        out[name] = (float(roc_auc_score(yt, ys))
                     if keep.sum() > 0 and len(np.unique(yt)) > 1 else float("nan"))
    return out


@torch.no_grad()
def run_validation(clip, tokenizer, device, val_loader, qformer_module=None,
                   use_anatomy_qformer=False, use_context=False, ctx_cfg=None,
                   assert_invariance=False):
    clip.eval()
    if qformer_module is not None:
        qformer_module.eval()
    pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)
    all_pred, all_true = [], []
    empty_sum = empty_n = 0.0
    for ct, _, _, labels, masks_fine, has_masks, _, ctx in tqdm.tqdm(val_loader, desc="Val-global", leave=False):
        ct = ct.to(device, non_blocking=True)
        with amp_context(device):
            feat_map = clip.visual_transformer.forward_spatial(ct)
            if use_context and qformer_module is not None:
                masks_fine = masks_fine.to(device, non_blocking=True)
                has_masks_d = has_masks.to(device, non_blocking=True)
                tok = qformer_module.context.tokenize(
                    tokenizer, ctx["indication"], ctx_cfg.max_ind_len, device)
                bundle = qformer_module.context(
                    tok["input_ids"], tok["attention_mask"],
                    ctx["age_band"].to(device), ctx["sex"].to(device),
                    age_years=ctx["age_years"].to(device),
                    age_mode=ctx_cfg.age_mode)
                out = qformer_module(feat_map, masks_fine, has_masks_d,
                                     context=bundle, return_parts=True)
                img_lat = out.z_final
                if out.empty is not None:
                    # A running mean over the WHOLE pass. build_role_mask's
                    # last_empty attribute is overwritten by every call, so the
                    # number the training log used to print was the last batch
                    # and nothing else.
                    empty_sum += float(out.empty.float().sum())
                    empty_n += float(out.empty.numel())
                if assert_invariance:
                    # The isolation claim, re-checked at this checkpoint rather
                    # than assumed to have survived training.
                    blank = qformer_module.context.tokenize(
                        tokenizer, [NO_INDICATION] * ct.shape[0],
                        ctx_cfg.max_ind_len, device)
                    b2 = qformer_module.context(
                        blank["input_ids"], blank["attention_mask"],
                        ctx["age_band"].to(device), ctx["sex"].to(device),
                        age_years=ctx["age_years"].to(device),
                        age_mode=ctx_cfg.age_mode)
                    o2 = qformer_module(feat_map, masks_fine, has_masks_d,
                                        context=b2, return_parts=True)
                    if not torch.equal(out.z_gen, o2.z_gen):
                        raise RuntimeError(
                            "Z_gen moved with the indication: max|d|="
                            f"{(out.z_gen - o2.z_gen).abs().max():.3e}")
            elif qformer_module is not None:
                if use_anatomy_qformer:
                    masks_fine = masks_fine.to(device, non_blocking=True)
                    has_masks_d = has_masks.to(device, non_blocking=True)
                    img_lat = qformer_module(feat_map, masks_fine, has_masks_d)
                else:
                    img_lat = qformer_module(feat_map)
            else:
                raw = clip.visual_transformer.global_pool(feat_map)
                img_lat = clip.to_visual_latent(raw)
            probs = prompt_probs(img_lat, pos_embs, neg_embs)
        all_pred.append(probs.float().cpu().numpy())
        all_true.append(labels.numpy())
    clip.train()
    if qformer_module is not None:
        qformer_module.train()
    if empty_n:
        print(f"[RAC] empty-region rate over the whole validation pass: "
              f"{empty_sum / empty_n:.4f} ({int(empty_sum):,}/{int(empty_n):,} "
              "slot-samples)", flush=True)
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    aucs = per_class_auc(pred, true)
    return float(np.nanmean(list(aucs.values()))), aucs


def _fusion_ratio(qformer_module) -> float:
    """||W_ind|| / ||W_gen||: how much of the fusion the conditioned path has taken.

    Reported every LOG_EVERY because it is the single number that says whether
    the conditioned half is contributing or running away. It starts at exactly 0
    -- the right block is initialised to zeros.
    """
    W = qformer_module.fusion.weight
    d = W.shape[1] // 2
    gen = float(W[:, :d].norm())
    return float(W[:, d:].norm()) / max(gen, 1e-8)


def lr_lambda(update):
    if update < WARMUP_UPDATES:
        return update / max(1, WARMUP_UPDATES)
    progress = (update - WARMUP_UPDATES) / max(1, TOTAL_UPDATES - WARMUP_UPDATES)
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _numbered_ckpts():
    ckpts = []
    for path in glob.glob(f"{RESULTS_DIR}/CTClip.*.pt"):
        m = re.search(r"CTClip\.(\d+)\.pt$", path)
        if m:
            ckpts.append((int(m.group(1)), path))
    return [p for _, p in sorted(ckpts)]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype()
    print(f"[RAC] device={device} amp={USE_AMP and device.type == 'cuda'} dtype={amp_dtype} results_dir={RESULTS_DIR}")
    print(f"[RAC] data_root={DATA_ROOT}")
    print(f"[RAC] train={DATA_TRAIN}\n[RAC] valid={DATA_VALID}")
    print(f"[RAC] masks train={MASK_TRAIN if os.path.isdir(MASK_TRAIN) else 'NONE'}")
    print(f"[RAC] masks valid={MASK_VALID if os.path.isdir(MASK_VALID) else 'NONE'}")
    print(f"[RAC] labels train={LABELS_TRAIN}\n[RAC] labels valid={LABELS_VALID}")
    print(f"[RAC] updates={TOTAL_UPDATES} batch={BATCH_SIZE} accum={ACCUM_STEPS} effective_batch={BATCH_SIZE * ACCUM_STEPS}")

    clip, tokenizer = build_model()
    clip = clip.to(device)
    clip.train()
    trainable, optimizer_groups = configure_trainable(clip)

    logit_scale = None
    if USE_LEARNABLE_TEMP:
        logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / TEMPERATURE), device=device))
        trainable.append(logit_scale)
        optimizer_groups.append({"params": [logit_scale], "lr": 1e-4, "weight_decay": 0.0})
        print(f"[RAC] Learnable temperature ENABLED init={TEMPERATURE} max_scale={LEARNABLE_TEMP_MAX_SCALE}")

    # T1.3 (FINAL_PIPELINE_2026-06-13.md): optionally defer optimizer/scheduler
    # construction until AFTER all optional modules add their param groups, so
    # QFormer / ITM / LM / salient receive proper warmup + cosine decay (not
    # constant LR). Env-gated, default OFF for v10/v11/v12/v13/v14 reproducibility.
    DEFER_OPT_BUILD = os.environ.get("RAC_OPT_DEFER", "0") == "1"
    if DEFER_OPT_BUILD:
        optimizer = None
        scheduler = None
        extra_groups: list[dict] = []
        print("[RAC] OPT_DEFER=1: optimizer/scheduler built after all optional modules.")
    else:
        optimizer = torch.optim.AdamW(optimizer_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        extra_groups = None  # not used in legacy path
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda" and amp_dtype == torch.float16)

    train_ds = RACDatasetV4(
        DATA_TRAIN,
        REPORTS_TRAIN,
        LABELS_TRAIN,
        mask_root=MASK_TRAIN if os.path.isdir(MASK_TRAIN) else None,
        region_cache_path=REGION_CACHE,
        is_train=True,
        limit=TRAIN_LIMIT,
        fail_fast=FAIL_FAST,
        volume_list_path=os.environ.get("RAC_VOLUME_LIST_TRAIN", ""),
        volume_exclude_path=os.environ.get("RAC_VOLUME_EXCLUDE", ""),
    )
    val_ds = RACDatasetV4(
        DATA_VALID,
        REPORTS_VALID,
        LABELS_VALID,
        mask_root=MASK_VALID if os.path.isdir(MASK_VALID) else None,
        is_train=False,
        limit=VAL_LIMIT,
        fail_fast=FAIL_FAST,
        volume_list_path=os.environ.get("RAC_VOLUME_LIST_VALID", ""),
        volume_exclude_path=os.environ.get("RAC_VOLUME_EXCLUDE", ""),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=rac_collate,
        drop_last=True,
        pin_memory=True,
        # persistent_workers keeps a COPY of the dataset in each worker, so
        # set_epoch() would never reach them and the indication-dropout mask
        # would be frozen for the whole run: a fixed 30% of volumes that never
        # see their indication, which is a different and worse experiment than
        # 30% resampled per epoch.
        persistent_workers=(NUM_WORKERS > 0 and not USE_CONTEXT_QFORMER),
        **_loader_kwargs(),
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=rac_collate, pin_memory=True, **_loader_kwargs())

    # AnTS-HardNeg init (env-gated). Loads hard-neg token packs + cooccur prior
    # and builds the loss module. Hard-neg sentences are projected through the
    # live text path each step so they track LoRA updates.
    if USE_TOPK_POOL:
        print(f"[RAC] Variant A TopK pool ENABLED: k={TOPK_K} focal_classes={sorted(FOCAL_CLASSES)}")
    qformer_module = None
    ctx_layout = None
    if USE_CONTEXT_QFORMER:
        # Built here, AFTER the seeding at import time and in the same position
        # in the RNG stream that AnatomyQFormer occupied, so qformer.queries
        # draws the same numbers as the reference run. Constructing any new
        # module *before* this point shifts the stream and silently changes the
        # initial query bank.
        anatomy_labels = list(range(1, 11))
        pathology_labels = [PATHOLOGY_FINE_ORGANS[name] for name in PATHOLOGIES]
        ctx_layout = SlotLayout.phase1(n_path=len(pathology_labels),
                                       n_anatomy=len(anatomy_labels))
        if QFORMER_NUM_QUERIES != ctx_layout.n_gen:
            raise RuntimeError(
                f"RAC_QFORMER_QUERIES={QFORMER_NUM_QUERIES} but the unconditioned "
                f"bank is {ctx_layout.n_gen} slots; the warm-start checkpoint and "
                "the step-0 identity both depend on these matching")
        qformer_module = ContextQFormer(
            anatomy_labels=anatomy_labels,
            pathology_labels=pathology_labels,
            layout=ctx_layout, cfg=CTX_CFG,
            dim=QFORMER_DIM, depth=QFORMER_DEPTH, num_heads=QFORMER_HEADS,
            image_dim=RACImageEncoder.FEAT_DIM,
        ).to(device)
        print(f"[RAC] Context Q-Former ENABLED: {ctx_layout.describe()} "
              f"c1={CTX_CFG.c1} c2={CTX_CFG.c2} selfattn={CTX_CFG.selfattn} "
              f"zind={CTX_CFG.zind_combine} dropout={CTX_CFG.ind_dropout}")
    elif USE_QFORMER or USE_ANATOMY_QFORMER:
        if USE_ANATOMY_QFORMER:
            # Map FINE_LABEL_NAMES ids 1..10 onto the anatomy queries; the
            # 18 pathology queries follow PATHOLOGIES order.
            anatomy_labels = list(range(1, 11))
            pathology_labels = [PATHOLOGY_FINE_ORGANS[name] for name in PATHOLOGIES]
            qformer_module = AnatomyQFormer(
                anatomy_labels=anatomy_labels,
                pathology_labels=pathology_labels,
                num_global=QFORMER_NUM_QUERIES - len(anatomy_labels) - len(pathology_labels),
                dim=QFORMER_DIM,
                depth=QFORMER_DEPTH,
                num_heads=QFORMER_HEADS,
                image_dim=RACImageEncoder.FEAT_DIM,
                pool=QFORMER_POOL,
            ).to(device)
            print(f"[RAC] Path 3 Anatomy-Bound Q-Former ENABLED: "
                  f"queries={qformer_module.qformer.num_queries} depth={QFORMER_DEPTH}")
        else:
            qformer_module = QFormer(
                num_queries=QFORMER_NUM_QUERIES,
                dim=QFORMER_DIM,
                depth=QFORMER_DEPTH,
                num_heads=QFORMER_HEADS,
                image_dim=RACImageEncoder.FEAT_DIM,
                pool=QFORMER_POOL,
            ).to(device)
            print(f"[RAC] Path 1 Q-Former ENABLED: queries={QFORMER_NUM_QUERIES} depth={QFORMER_DEPTH}")

    # Shared tail: registration, param groups, and -- for the context bank --
    # the split into inherited and new parameters.
    ctx_params: list = []
    if qformer_module is not None:
        for p in qformer_module.parameters():
            p.requires_grad = True
        qf_params = list(qformer_module.parameters())
        groups_to_add = []
        if USE_CONTEXT_QFORMER:
            # The new modules produce ZERO output at step 0 but NOT zero
            # gradient: dL/dg_x = <a, dL/dq> is nonzero, and so is dL/dW_right.
            # Under a single global clip_grad_norm_ their contribution inflates
            # the total norm, so every inherited parameter takes a SMALLER step
            # than the reference run from update 1 onwards. The forward identity
            # survives that; the trajectory does not. Clipping the two sets
            # separately is what keeps "step 0 is ARC-CT" from decaying into
            # "step 0 is ARC-CT and then something slower".
            ctx_mods = [qformer_module.context, qformer_module.conditioner,
                        qformer_module.relevance, qformer_module.fusion]
            ctx_ids = {id(p) for m in ctx_mods for p in m.parameters()}
            ctx_params = [p for p in qf_params if id(p) in ctx_ids]
            inherited = [p for p in qf_params if id(p) not in ctx_ids]
            # e_int carries 10x weight decay: the interaction table starts at
            # exactly zero and should stay there unless it pays for itself.
            e_int_id = {id(qformer_module.context.e_int)}
            groups_to_add.append({"params": inherited, "lr": VISION_LR,
                                  "weight_decay": WEIGHT_DECAY})
            groups_to_add.append({"params": [p for p in ctx_params if id(p) not in e_int_id],
                                  "lr": VISION_LR, "weight_decay": WEIGHT_DECAY})
            groups_to_add.append({"params": [qformer_module.context.e_int],
                                  "lr": VISION_LR, "weight_decay": WEIGHT_DECAY * 10.0})
            print(f"[RAC] Q-Former params: inherited {sum(p.numel() for p in inherited):,} "
                  f"| context {sum(p.numel() for p in ctx_params):,} "
                  f"(clipped separately)")
        else:
            groups_to_add.append({"params": qf_params, "lr": VISION_LR,
                                  "weight_decay": WEIGHT_DECAY})
        trainable.extend(qf_params)
        for _group in groups_to_add:
            if not _group["params"]:
                continue
            if DEFER_OPT_BUILD:
                extra_groups.append(_group)
            else:
                optimizer.add_param_group(_group)
        n_qf = sum(p.numel() for p in qf_params)
        print(f"[RAC] Q-Former trainable params: {n_qf:,}")

    if DEFER_OPT_BUILD:
        all_groups = optimizer_groups + extra_groups
        optimizer = torch.optim.AdamW(all_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[RAC] OPT_DEFER finalized: {len(all_groups)} param groups under one scheduler.")

    update_step = 0
    micro_step = 0
    best_auc = 0.0
    best_step = 0
    patience = 0
    existing = _numbered_ckpts()
    best_ckpt = f"{RESULTS_DIR}/CTClip.best.pt"
    resume_from = None
    if os.path.isfile(best_ckpt):
        resume_from = best_ckpt
    elif existing:
        resume_from = existing[-1]
    if resume_from is not None:
        pkg = torch.load(resume_from, map_location=device)
        missing, unexpected = clip.load_state_dict(pkg["model"], strict=False)
        optimizer.load_state_dict(pkg["optimizer"])
        if int(os.environ.get("RAC_RESET_SCHED", "0")):
            print(f"[RAC] RAC_RESET_SCHED=1 -> skip scheduler.load_state_dict; cosine restarts from epoch=0 (full LR)")
        else:
            scheduler.load_state_dict(pkg["scheduler"])
        if "scaler" in pkg:
            scaler.load_state_dict(pkg["scaler"])
        if logit_scale is not None and "logit_scale" in pkg:
            with torch.no_grad():
                logit_scale.data.copy_(pkg["logit_scale"].to(logit_scale.device))
            cur_tau = float(1.0 / logit_scale.detach().exp().clamp(max=LEARNABLE_TEMP_MAX_SCALE))
            print(f"[RAC] logit_scale restored, current tau={cur_tau:.4f}")
        update_step = int(pkg.get("update_step", pkg.get("step", 0)))
        best_auc = float(pkg.get("best_auc", 0.0))
        # Resume restored model/optimizer/scheduler but not the Q-Former, while
        # it DID restore best_auc. After a preemption the Q-Former would come
        # back random and could never beat the restored best, so the best
        # checkpoint would never be rewritten again - invisible without a
        # weight diff.
        if qformer_module is not None:
            if "qformer_module" not in pkg:
                raise RuntimeError(
                    f"[RAC] resume checkpoint {resume_from} has no 'qformer_module'")
            qm, qu = qformer_module.load_state_dict(pkg["qformer_module"], strict=False)
            print(f"[RAC] Resumed Q-Former: missing={len(qm)} unexpected={len(qu)}", flush=True)
        print(f"[RAC] Resumed {resume_from} update={update_step} best_auc={best_auc:.4f} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        warm = os.environ.get("RAC_WARM_START_CKPT", "").strip()
        if warm and os.path.isfile(warm):
            pkg = torch.load(warm, map_location=device)
            missing, unexpected = clip.load_state_dict(pkg.get("model", pkg), strict=False)
            print(f"[RAC] Warm-started CTCLIP from {warm}: missing={len(missing)} unexpected={len(unexpected)}")
            # The Q-Former is a separate module and was NOT covered by the line
            # above: warm-starting only clip left ~38M Q-Former parameters at
            # random init while the loss curves looked healthy because the
            # backbone was warm. A remapped checkpoint (30 -> 39 query rows) was
            # simply ignored. Load it explicitly, and fail loudly if it cannot
            # be loaded rather than training from noise.
            if qformer_module is not None and "qformer_module" in pkg:
                qsd = pkg["qformer_module"]
                qrows = qsd.get("qformer.queries")
                want = qformer_module.qformer.queries.shape
                if USE_CONTEXT_QFORMER:
                    # A 39-row ARC-CT bank maps onto the 40-row context bank by
                    # appending one row: the conditioned pathology slots are the
                    # SAME parameters read twice, so Phase 1 adds exactly one new
                    # query. No external remap script is involved.
                    qm, qu = qformer_module.load_arcct_state_dict(qsd)
                else:
                    if qrows is not None and tuple(qrows.shape) != tuple(want):
                        raise RuntimeError(
                            f"[RAC] Q-Former query shape {tuple(qrows.shape)} in {warm} "
                            f"does not match the model's {tuple(want)}. Remap the "
                            f"checkpoint (tools/remap_ckpt_schema.py) or fix "
                            f"RAC_QFORMER_QUERIES / RAC_SCHEMA.")
                    qm, qu = qformer_module.load_state_dict(qsd, strict=False)
                print(f"[RAC] Warm-started Q-Former from {warm}: "
                      f"queries={tuple(qrows.shape) if qrows is not None else None} "
                      f"missing={len(qm)} unexpected={len(qu)}", flush=True)
            elif qformer_module is not None:
                raise RuntimeError(
                    f"[RAC] {warm} has no 'qformer_module'; the Q-Former would "
                    f"train from random init. Refusing to start silently.")

    def save_ckpt(name):
        extra = {}
        if qformer_module is not None:
            extra["qformer_module"] = qformer_module.state_dict()
        if logit_scale is not None:
            extra["logit_scale"] = logit_scale.detach().cpu()
        torch.save(
            {
                "model": clip.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "update_step": update_step,
                "best_auc": best_auc,
                **extra,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "accum_steps": ACCUM_STEPS,
                    "total_updates": TOTAL_UPDATES,
                    "vision_lr": VISION_LR,
                    "text_lr": TEXT_LR,
                    "clip_weight": CLIP_WEIGHT,
                    "cls_weight": CLS_WEIGHT,
                    "align_weight": ALIGN_WEIGHT,
                    "fn_weight": FN_WEIGHT,
                    "max_text_len": MAX_TEXT_LEN,
                    "use_lora": USE_LORA,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "organ_labels": ORGAN_LABELS,
                    # Written so evaluate.py never has to infer the bank shape
                    # from the tensor: a 40-row context bank and a 3-global
                    # AnatomyQFormer have identical shapes, and guessing between
                    # them evaluates the wrong model with no error at all.
                    **({"query_layout": qformer_module.query_layout()}
                       if USE_CONTEXT_QFORMER and qformer_module is not None else {}),
                },
            },
            f"{RESULTS_DIR}/{name}.pt",
        )
        print(f"[RAC] Saved {RESULTS_DIR}/{name}.pt")

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)
    if USE_CONTEXT_QFORMER and qformer_module is not None:
        # The relevance head compares indication text against class-prompt text,
        # so BOTH sides come from the frozen indication tower rather than from
        # the trained visual-alignment projection. That makes these embeddings a
        # constant of the run -- unlike pos_embs, which is re-encoded every
        # PROMPT_CACHE_EVERY updates as the text LoRA drifts -- so this is
        # installed once here and never goes stale on resume.
        qformer_module.relevance.set_class_embeddings(
            qformer_module.context.encode_class_prompts(tokenizer, list(PATHOLOGIES)))
        print(f"[RAC] relevance class embeddings installed: "
              f"{tuple(qformer_module.relevance.class_emb.shape)} (frozen basis)")
    log = {"clip": 0.0, "cls": 0.0, "align": 0.0, "ptok": 0.0, "ptok_ind": 0.0, "cf": 0.0, "n": 0.0, "mask": 0.0}
    stop_early = False
    t0 = time.time()
    pbar = tqdm.tqdm(initial=update_step, total=TOTAL_UPDATES, desc="RAC-updates")

    while update_step < TOTAL_UPDATES:
        try:
            ct, texts, findings, labels, masks_fine, has_masks, accessions, ctx = next(train_iter)
        except StopIteration:
            # One pass finished. Advancing the epoch is what re-rolls the
            # indication dropout; see set_epoch().
            train_ds.set_epoch(getattr(train_ds, "epoch", 0) + 1)
            train_iter = iter(train_loader)
            ct, texts, findings, labels, masks_fine, has_masks, accessions, ctx = next(train_iter)

        ct = ct.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks_fine = masks_fine.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        with amp_context(device):
            feat_map = clip.visual_transformer.forward_spatial(ct)
            qf_tokens = None
            ctx_out = None
            if USE_CONTEXT_QFORMER and qformer_module is not None:
                ctx_tok = qformer_module.context.tokenize(
                    tokenizer, ctx["indication"], CTX_CFG.max_ind_len, device)
                ctx_bundle = qformer_module.context(
                    ctx_tok["input_ids"], ctx_tok["attention_mask"],
                    ctx["age_band"].to(device), ctx["sex"].to(device),
                    age_years=ctx["age_years"].to(device),
                    age_mode=CTX_CFG.age_mode)
                ctx_out = qformer_module(
                    feat_map, masks_fine, has_masks, context=ctx_bundle,
                    return_parts=True)
                img_lat, qf_tokens = ctx_out.z_final, ctx_out.tokens
            elif qformer_module is not None:
                if USE_ANATOMY_QFORMER:
                    # Anatomy-Bound Q-Former needs the TS mask + has_mask flags.
                    img_lat, qf_tokens = qformer_module(
                        feat_map, masks_fine, has_masks, return_tokens=True
                    )
                else:
                    img_lat, qf_tokens = qformer_module(feat_map, return_tokens=True)
            else:
                raw = clip.visual_transformer.global_pool(feat_map)
                img_lat = clip.to_visual_latent(raw)
            txt_lat = encode_texts(clip, tokenizer, texts, device)
            if logit_scale is not None:
                temp_now = 1.0 / logit_scale.exp().clamp(max=LEARNABLE_TEMP_MAX_SCALE)
            else:
                temp_now = TEMPERATURE
            loss_clip = soft_clip_infonce(img_lat, txt_lat, labels, temperature=temp_now)
            loss_cls = prompt_cls_loss(img_lat, pos_embs, neg_embs, labels, temperature=temp_now)

            loss_align = torch.tensor(0.0, device=device)
            loss_pertoken = torch.tensor(0.0, device=device)
            loss_ptok_ind = torch.tensor(0.0, device=device)
            loss_cf = torch.tensor(0.0, device=device)

            def _pertoken_terms(path_tokens):
                """Per (sample, class) cross-entropy against pos/neg prompts.

                Returns the per-cell loss and its labelled mask, so a caller can
                either average it (the inherited term) or weight it (C2).
                """
                path_lat = F.normalize(path_tokens, dim=-1)
                sims_pos = (path_lat * pos_embs.unsqueeze(0)).sum(dim=-1)
                sims_neg = (path_lat * neg_embs.unsqueeze(0)).sum(dim=-1)
                logits_pt = torch.stack([sims_pos, sims_neg], dim=-1) / temp_now
                log_p_pt = F.log_softmax(logits_pt, dim=-1)
                # NaN in, NaN out, with nothing raised. Mask per cell and
                # average over labeled cells only, so the loss magnitude does
                # not depend on how many cells are unlabeled.
                _lab_pt = labels.float()
                _m_pt = torch.isfinite(_lab_pt)
                _lab_pt = torch.nan_to_num(_lab_pt, nan=0.0)
                target_pt = torch.stack([_lab_pt, 1.0 - _lab_pt], dim=-1)
                return -(target_pt * log_p_pt).sum(dim=-1), _m_pt

            if (PERTOKEN_NCE_WEIGHT > 0.0 and qf_tokens is not None
                    and qformer_module is not None
                    and (USE_ANATOMY_QFORMER or USE_CONTEXT_QFORMER)):
                if USE_CONTEXT_QFORMER:
                    # Named access, not arithmetic. With the unconditioned-first
                    # layout qf_tokens[:, A:A+P] still happens to land on the
                    # general bank, but that is a coincidence of the slot order
                    # rather than a statement of intent.
                    gen_tok = qf_tokens.index_select(1, qformer_module.path_gen_index)
                else:
                    A_q = len(qformer_module.anatomy_labels)
                    P_q = len(qformer_module.pathology_labels)
                    gen_tok = qf_tokens[:, A_q:A_q + P_q, :]
                _per_pt, _m_pt = _pertoken_terms(gen_tok)
                # The UNCONDITIONED bank keeps ARC-CT's unweighted loss. The
                # adult gate and the step-0 story both rest on this term being
                # numerically what it was before.
                loss_pertoken = (_per_pt * _m_pt).sum() / _m_pt.sum().clamp(min=1)

                if USE_CONTEXT_QFORMER and CTX_PTOK_WEIGHT > 0.0 and ctx_out is not None:
                    cond_tok = qf_tokens.index_select(1, qformer_module.path_cond_index)
                    _per_c, _m_c = _pertoken_terms(cond_tok)
                    w_c = ctx_out.w.to(_per_c.dtype) * _m_c
                    # Normalised by sum(w): un-normalised, every per-class loss
                    # is >= 0 so dL/dbeta > 0 always and beta is driven to its
                    # floor from step 1.
                    loss_ptok_ind = ((w_c * _per_c).sum(dim=-1)
                                     / w_c.sum(dim=-1).clamp(min=1e-6)).mean()
            n_masked = int(has_masks.sum().item())
            if n_masked > 0 and ORGAN_LABELS:
                midx = has_masks.nonzero(as_tuple=True)[0]
                feat_m = feat_map[midx]
                masks_m = masks_fine[midx]
                lab_m = labels[midx]
                find_m = [findings[i] for i in midx.tolist()]
                acc_m = [accessions[i] for i in midx.tolist()]
                organ_raw, organ_valid = clip.visual_transformer.pool_organ_masks(feat_m, masks_m, ORGAN_LABELS)
                Bm, Om, C = organ_raw.shape
                organ_lat = clip.to_visual_latent(organ_raw.reshape(Bm * Om, C)).reshape(Bm, Om, -1)
                organ_lat_focal = None
                if USE_TOPK_POOL:
                    organ_raw_topk, _ = clip.visual_transformer.pool_organ_masks_topk(
                        feat_m, masks_m, ORGAN_LABELS, k=TOPK_K
                    )
                    organ_lat_focal = clip.to_visual_latent(
                        organ_raw_topk.reshape(Bm * Om, C)
                    ).reshape(Bm, Om, -1)
                align_terms = []
                for oi, organ_label in enumerate(ORGAN_LABELS):
                    valid = organ_valid[:, oi]
                    if int(valid.sum().item()) < 2:
                        continue
                    idxs = valid.nonzero(as_tuple=True)[0]
                    path_indices = ORGAN_TO_PATHIDX[organ_label]
                    region_texts = [
                        region_text_from_cache(train_ds.region_cache, acc_m[j], organ_label, find_m[j])
                        for j in idxs.tolist()
                    ]
                    region_lat = encode_texts(clip, tokenizer, region_texts, device)
                    if USE_FOCAL_ALIGN and organ_lat_focal is not None:
                        # Route each class to the pooling that suits it: focal
                        # classes to top-K, the rest to the mean. Splitting the
                        # term rather than replacing it keeps the diffuse
                        # classes (effusion, opacity of a whole lobe) on the
                        # operator they are actually well served by.
                        foc = [k for k, pi in enumerate(path_indices) if pi in FOCAL_PATHIDX]
                        dif = [k for k, pi in enumerate(path_indices) if pi not in FOCAL_PATHIDX]
                        if dif:
                            di = [path_indices[k] for k in dif]
                            align_terms.append(soft_clip_infonce(
                                organ_lat[idxs, oi], region_lat,
                                lab_m[idxs][:, di], temperature=temp_now))
                        if foc:
                            fi = [path_indices[k] for k in foc]
                            align_terms.append(soft_clip_infonce(
                                organ_lat_focal[idxs, oi], region_lat,
                                lab_m[idxs][:, fi], temperature=temp_now))
                    else:
                        align_terms.append(soft_clip_infonce(organ_lat[idxs, oi], region_lat, lab_m[idxs][:, path_indices], temperature=temp_now))
                if align_terms:
                    loss_align = torch.stack(align_terms).mean()

                # AnTS-HardNeg term (env-gated). Operates on the SAME region
                # pools as region_align, plus a per-class text bank and
                # FaNe-reweighted within-batch visual hard negatives.
            # -- counterfactual consistency, on the PREDICTIONS -------------
            # Isolation proves dZ_gen/dI = 0, but says nothing about Z_final:
            # the right block of the fusion is unconstrained after step 0 and
            # Z_ind depends entirely on the indication. The safety claim is
            # about the prediction -- "the pneumothorax must not disappear
            # because the scan was ordered for pneumonia" -- so the penalty
            # belongs on the logits of the classes this indication considers
            # irrelevant, not on Z_gen where it would be identically zero.
            if (USE_CONTEXT_QFORMER and CF_WEIGHT > 0.0 and ctx_out is not None
                    and any(ctx["indication_cf"])):
                cf_idx = [i for i, t in enumerate(ctx["indication_cf"]) if t]
                # Capped. The counterfactual pass runs a SECOND Q-Former forward
                # with its own graph, which roughly doubles the bank's
                # activations, and at batch 20 over 67 slots that is what pushed
                # an A40 over. Capping the sub-batch is a budget on an auxiliary
                # term; lowering RAC_BATCH_SIZE instead would shrink the
                # in-batch negative pool that soft_clip_infonce depends on and
                # make the context rungs incomparable to ct_only.
                if len(cf_idx) > CF_MAX_ROWS:
                    cf_idx = cf_idx[:CF_MAX_ROWS]
                if cf_idx:
                    sel = torch.tensor(cf_idx, device=device)
                    cf_txt = [ctx["indication_cf"][i] for i in cf_idx]
                    cf_tok = qformer_module.context.tokenize(
                        tokenizer, cf_txt, CTX_CFG.max_ind_len, device)
                    cf_bundle = qformer_module.context(
                        cf_tok["input_ids"], cf_tok["attention_mask"],
                        ctx["age_band"].to(device)[sel], ctx["sex"].to(device)[sel],
                        age_years=ctx["age_years"].to(device)[sel],
                        age_mode=CTX_CFG.age_mode)
                    # feat_map is REUSED: the image did not change, only H_C did.
                    # Re-running forward_spatial would double the step cost for
                    # an identical tensor.
                    cf_out = qformer_module(
                        feat_map[sel], masks_fine[sel], has_masks[sel],
                        context=cf_bundle, return_parts=True)
                    p_true = prompt_probs(img_lat[sel], pos_embs, neg_embs, temp_now)
                    p_cf = prompt_probs(cf_out.z_final, pos_embs, neg_embs, temp_now)
                    # With C2 off there is no relevance signal, so there is no
                    # basis for calling any class irrelevant -- and the honest
                    # default is the STRONGER constraint: hold every class
                    # stable under a mismatched indication. Indexing r here
                    # without the guard is what crashed the three C2-off rungs.
                    if ctx_out.r is None:
                        low = torch.ones_like(p_true)
                    else:
                        low = (ctx_out.r[sel] < CF_TAU).to(p_true.dtype)
                    loss_cf = ((p_true - p_cf).pow(2) * low).sum(dim=-1).div(
                        low.sum(dim=-1).clamp(min=1.0)).mean()

            total = (CLIP_WEIGHT * loss_clip + CLS_WEIGHT * loss_cls
                     + ALIGN_WEIGHT * loss_align
                     + PERTOKEN_NCE_WEIGHT * loss_pertoken
                     + CTX_PTOK_WEIGHT * loss_ptok_ind
                     + CF_WEIGHT * loss_cf)
            total_to_backprop = total / ACCUM_STEPS

        if scaler.is_enabled():
            scaler.scale(total_to_backprop).backward()
        else:
            total_to_backprop.backward()
        micro_step += 1

        log["clip"] += float(loss_clip.detach().cpu())
        log["cls"] += float(loss_cls.detach().cpu())
        log["align"] += float(loss_align.detach().cpu())
        log["ptok"] += float(loss_pertoken.detach().cpu()) if isinstance(loss_pertoken, torch.Tensor) else 0.0
        log["ptok_ind"] += float(loss_ptok_ind.detach().cpu())
        log["cf"] += float(loss_cf.detach().cpu())
        log["mask"] += n_masked
        log["n"] += 1

        if micro_step % ACCUM_STEPS != 0:
            continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if ctx_params:
            # Two clips, not one. The context modules emit zero at step 0 but
            # NOT zero gradient, so a single global norm would be inflated by
            # parameters that contribute nothing to the output -- and every
            # inherited parameter would then take a smaller step than the
            # reference run from update 1.
            _ctx_ids = {id(p) for p in ctx_params}
            nn.utils.clip_grad_norm_([p for p in trainable if id(p) not in _ctx_ids],
                                     MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(ctx_params, MAX_GRAD_NORM)
        else:
            nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        update_step += 1
        pbar.update(1)

        if update_step % PROMPT_CACHE_EVERY == 0:
            pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)

        if update_step % LOG_EVERY == 0:
            lr_vision = optimizer.param_groups[0]["lr"]
            lr_text = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
            denom = max(log["n"], 1.0)
            # Only the four losses this release actually computes are logged.
            # Earlier revisions also printed ants/sal/itm/lm/siglip/anat/stk/
            # asup/smil/imp, but those terms were dropped along with their
            # ablation code while the print was left behind - which made the
            # first LOG_EVERY boundary raise KeyError.
            print(
                f"[RAC] update={update_step:5d} micro={micro_step:7d} "
                f"clip={log['clip']/denom:.4f} cls={log['cls']/denom:.4f} "
                f"align={log['align']/denom:.4f} ptok={log['ptok']/denom:.4f} "
                + (f"ptok_ind={log['ptok_ind']/denom:.4f} cf={log['cf']/denom:.4f} "
                   f"beta={float(qformer_module.relevance.beta):.4f} "
                   f"|Wi|/|Wg|={_fusion_ratio(qformer_module):.3f} "
                   if USE_CONTEXT_QFORMER and qformer_module is not None else "") +
                f"masked/batch={log['mask']/denom:.2f} "
                f"lr={lr_vision:.2e}/{lr_text:.2e} t={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
            log = {"clip": 0.0, "cls": 0.0, "align": 0.0, "ptok": 0.0, "ptok_ind": 0.0, "cf": 0.0, "n": 0.0, "mask": 0.0}  # noqa: E501

        if update_step % VAL_EVERY == 0:
            mean_auc, per_auc = run_validation(
                clip, tokenizer, device, val_loader,
                qformer_module=qformer_module,
                use_anatomy_qformer=USE_ANATOMY_QFORMER,
                use_context=USE_CONTEXT_QFORMER, ctx_cfg=CTX_CFG,
                assert_invariance=CTX_CFG.assert_invariance,
            )
            pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)
            # Report how often each anatomy query actually had a region to attend to.

            # Without this an empty region is indistinguishable from a working one:

            # build_role_mask silently unrestricts it and training carries on.

            _er = getattr(build_role_mask, "last_empty", None)

            if _er is not None:

                _a = _er[:len(ANATOMY_LABELS)] if "ANATOMY_LABELS" in dir() else _er[:10]

                # last_empty is set on every build_role_mask call, so this is the LAST
# batch, not an average. In the combined run the valid list is sorted
# with pediatric first and adult last, so the final batch is all-adult,
# has no masks, and prints 1.00 for every region - which reads exactly
# like total masking failure and is not.
                print("[RAC] empty-region rate (SON BATCH) per anatomy query: "

                      + " ".join(f"{i+1}:{float(v):.2f}" for i, v in enumerate(_a)), flush=True)

            print(f"[RAC] update={update_step} global_val_auc={mean_auc:.4f} best={best_auc:.4f}", flush=True)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_step = update_step
                patience = 0
                save_ckpt("CTClip.best")
                save_ckpt(f"CTClip.{update_step}")
                with open(f"{RESULTS_DIR}/best_auc_update{update_step}.txt", "w") as f:
                    for name in PATHOLOGIES:
                        f.write(f"{name:45s}: {per_auc[name]:.4f}\n")
                    f.write(f"{'Mean AUC':45s}: {mean_auc:.4f}\n")
            else:
                patience += 1
                if EARLY_STOP_PATIENCE > 0 and update_step >= EARLY_STOP_MIN_UPDATES and patience >= EARLY_STOP_PATIENCE:
                    print(f"[RAC] EARLY STOP update={update_step} best={best_auc:.4f} at update={best_step}", flush=True)
                    stop_early = True

        if update_step % SAVE_EVERY == 0:
            save_ckpt(f"CTClip.{update_step}")
        if stop_early:
            break

    pbar.close()
    save_ckpt(f"CTClip.{update_step}")
    print(f"[RAC] Done. Best global val AUC={best_auc:.4f} at update={best_step}")


if __name__ == "__main__":
    main()
