"""
HLIP zero-shot evaluation on the 23-class BCH pediatric chest-CT benchmark.

This is HLIP/src/hlip_test/zeroshot_ct_rate.py adapted to the pediatric data:
  * the model is HLIP's UNMODIFIED definition (src/hlip/visual_encoder.py + model_configs),
    loaded through open_clip.create_model_and_transforms exactly as in the reference script;
  * image loading / windowing / crop / normalisation is byte-for-byte the reference recipe
    (window [-1150, 350], clip to [0, 1], pad+centre-crop to 112x336x336, Normalize with the
    mean of IMAGENET_DEFAULT_MEAN/STD, then a leading scan dimension);
  * only the file layout (flat <root>/<split>/<volume>.pt) and the prompt metadata
    (23 pediatric classes) differ.

Outputs an npz with:
    pred    float32 (n, 23)  positive-class softmax probability
    true    int32   (n, 23)
    classes <U..    (23,)
    volumes <U..    (n,)     volume names, in row order
"""

import os
import sys
import json
import math
import random
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

HLIP_SRC = os.environ.get('HLIP_SRC', '/home/ch278233/BENCHMARK/HLIP/src')
sys.path.insert(0, HLIP_SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

from hlip import visual_encoder  # noqa: F401  (registers the timm models)
from zeroshot_metadata_peds23 import CLASSNAMES, ORGANS, TEMPLATES, PROMPTS, VOLUME_PROMPTS


def get_args_parser():
    parser = argparse.ArgumentParser('HLIP pediatric zero-shot', add_help=False)
    parser.add_argument('--model', default='clip_vit_base_singlescan_h2_token2744', type=str)
    parser.add_argument('--use-cxr-bert', default=False, action='store_true')
    parser.add_argument('--resume', default='/temp_work/ch278233/PEDS_BENCH/weights/hlip/chestct_clip_vit_base_singlescan_h2_token2744.pt', type=str)
    parser.add_argument('--model-configs', default=os.path.join(HLIP_SRC, 'hlip', 'model_configs'), type=str)

    parser.add_argument('--data-root', default='/temp_work/ch278233/PEDS_BENCH/hlip/pt/', type=str)
    parser.add_argument('--split', default='valid', type=str)
    parser.add_argument('--labels', default='/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv', type=str)
    parser.add_argument('--input-info', nargs='+', default=["-1150", "350", "crop"])
    parser.add_argument('--zeroshot-template', default='volume', type=str)

    parser.add_argument('--out-npz', default='/temp_work/ch278233/PEDS_BENCH/preds/hlip_peds_zeroshot_preds.npz', type=str)
    parser.add_argument('--results-dir', default='/temp_work/ch278233/PEDS_BENCH/hlip/results/', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--limit', default=0, type=int, help='debug: only the first N volumes')
    parser.add_argument('--allow-missing', default=False, action='store_true',
                        help='skip volumes whose .pt is absent instead of failing')
    return parser


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class ZeroShotDataset(Dataset):
    """Identical preprocessing to zeroshot_ct_rate.py::get_data.ZeroShotDataset."""

    def __init__(self, root, split, input_filename, input_info, limit=0, allow_missing=False):
        self.cts = []
        missing = 0
        df = pd.read_csv(input_filename)
        for _, row in df.iterrows():
            recon = str(row['VolumeName']).split('.')[0]        # ped_00002_1.nii.gz -> ped_00002_1
            path = os.path.join(root, split, recon + '.pt')
            if not os.path.exists(path):
                missing += 1
                if allow_missing:
                    continue
            self.cts.append((path, row[CLASSNAMES].astype(int).tolist()))
        if limit:
            self.cts = self.cts[:limit]
        self.missing = missing
        self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx):
        recon, target = self.cts[idx]

        # .float() before windowing: the CT-RATE pipeline stores float32, we store float16
        # (to save disk), so we promote first and the windowing arithmetic is then identical
        # to the reference. Only the stored HU value carries float16 quantisation (<= 1 HU).
        img = torch.load(recon, weights_only=True).float()
        img = (img - self.input_info[0]) / (self.input_info[1] - self.input_info[0])
        img = torch.clip(img, 0., 1.)
        img = img[None, ...].float()

        if self.input_info[2] == "crop":
            _, d, h, w = img.shape
            pad_d = max(112 - d, 0)
            pad_h = max(336 - h, 0)
            pad_w = max(336 - w, 0)
            pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
            pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
            pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
            img = torch.nn.functional.pad(
                img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
                mode='constant', value=0
            ).squeeze(0)

            _, d, h, w = img.shape
            start_d = (d - 112) // 2
            start_h = (h - 336) // 2
            start_w = (w - 336) // 2
            img = img[:, start_d:start_d + 112, start_h:start_h + 336, start_w:start_w + 336]

        elif self.input_info[2] == "resize":
            img = torch.nn.functional.interpolate(img[None, ...], size=(112, 336, 336), mode='trilinear').squeeze(0)

        else:
            raise NotImplementedError

        normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
        img = normalizer(img)

        return recon, img[None, ...], torch.as_tensor(target)


def build_model(args):
    for _c in os.listdir(args.model_configs):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join(args.model_configs, _c), 'r') as f:
                _MODEL_CONFIGS[_m] = json.load(f)

    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)

    if args.use_cxr_bert:
        from transformers import AutoModel
        cxr_bert = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True).bert
        cxr_bert.to(device=args.device)
        model.text.transformer = cxr_bert

    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}

    msd = model.state_dict()
    n_match = sum(1 for k, v in sd.items() if k in msd and msd[k].shape == v.shape)
    print(f'[checkpoint] tensors in ckpt: {len(sd)} | tensors in model: {len(msd)} | name+shape matches: {n_match}', flush=True)
    if n_match != len(sd) or n_match != len(msd):
        print('[checkpoint] missing from ckpt:', [k for k in msd if k not in sd][:20], flush=True)
        print('[checkpoint] unexpected in ckpt:', [k for k in sd if k not in msd][:20], flush=True)
        raise SystemExit('ARCHITECTURE MISMATCH -- refusing to force-load. See lists above.')

    model.load_state_dict(sd)  # strict
    print('[checkpoint] strict load: all keys matched', flush=True)
    return model, n_match


def main(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    prompts = dict(PROMPTS)
    if args.zeroshot_template != 'organ':
        prompts.update(VOLUME_PROMPTS)

    random_seed(0, 0)

    model, n_match = build_model(args)
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    model.eval()

    dataset = ZeroShotDataset(args.data_root, args.split, args.labels, args.input_info,
                              limit=args.limit, allow_missing=args.allow_missing)
    print(f'[data] {len(dataset)} volumes ({dataset.missing} missing .pt files)', flush=True)
    if dataset.missing and not args.allow_missing:
        raise SystemExit(f'{dataset.missing} preprocessed volumes are missing under '
                         f'{os.path.join(args.data_root, args.split)} -- refusing to run a partial evaluation. '
                         f'Re-run preprocessing, or pass --allow-missing deliberately.')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=True, sampler=None, drop_last=False)

    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    with autocast():
        classifier = {}
        for key in CLASSNAMES:
            templates = TEMPLATES[ORGANS[key]] if args.zeroshot_template == 'organ' else TEMPLATES[args.zeroshot_template]
            print(f'[prompt] {key}: ' + ' | '.join(t(p) for p in prompts[key] for t in templates), flush=True)
            classifier[key] = build_zero_shot_classifier(
                model, tokenizer=tokenizer, classnames=prompts[key], templates=templates,
                num_classes_per_batch=None, device=device, use_tqdm=False,
            )

    preds, trues, vols = [], [], []
    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader)):
            recon, image, target = batch
            image = image.to(device=device, dtype=input_dtype)
            row = []
            with autocast():
                output = model(image=image)
                image_features = output['image_features']
                logit_scale = output['logit_scale']
                for key in CLASSNAMES:
                    logits_per_image = logit_scale * image_features @ classifier[key]
                    logits_per_image = logits_per_image.softmax(dim=1)
                    row.append(logits_per_image[0, 1].cpu().float().item())
            preds.append(row)
            trues.append(target[0].cpu().numpy().astype(np.int32).tolist())
            vols.append(os.path.basename(recon[0])[:-3])

    pred = np.asarray(preds, dtype=np.float32)
    true = np.asarray(trues, dtype=np.int32)
    classes = np.asarray(CLASSNAMES)
    volumes = np.asarray(vols)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(args.out_npz, pred=pred, true=true, classes=classes, volumes=volumes)
    print(f'[saved] {args.out_npz}  pred{pred.shape} true{true.shape} classes{classes.shape}', flush=True)

    # convenience metrics; never allowed to invalidate the saved predictions
    try:
        from sklearn.metrics import roc_auc_score
        aucs = {}
        for i, c in enumerate(CLASSNAMES):
            y = true[:, i]
            aucs[c] = float(roc_auc_score(y, pred[:, i])) if 0 < y.sum() < len(y) else None
        valid = [v for v in aucs.values() if v is not None]
        aucs['* mean auc'] = float(np.mean(valid)) if valid else None
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, f'auc_{args.zeroshot_template}.json'), 'w') as f:
            json.dump({'n': int(len(true)), 'tensor_matches': int(n_match), 'auc': aucs}, f, indent=2)
        print('[auc] mean =', aucs['* mean auc'], flush=True)
    except Exception as e:
        print('[warn] metric computation failed (predictions are saved):', e, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HLIP pediatric zero-shot', parents=[get_args_parser()])
    main(parser.parse_args())
