"""
HLIP retrieval latents for the pediatric validation split.

Produces /temp_work/ch278233/PEDS_BENCH/retrieval/hlip_valid.npz via
peds_report_text.write_retrieval_npz (which adds labels / accessions / text_ok / classes).

  img_lat : HLIP image_features exactly as the zero-shot script obtains them
            (open_clip CLIP.forward -> encode_image(normalize=True)), re-L2-normalised (no-op)
  txt_lat : HLIP text tower on the canonical REPORT string from peds_report_text.texts_for
            (open_clip encode_text(normalize=True)), same embedding space

No metrics are computed here.  Model construction and image preprocessing are imported
unchanged from zeroshot_peds23_hlip.py (350/350 strict checkpoint match).
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_peds23_hlip import build_model, ZeroShotDataset, get_args_parser as zs_parser, HLIP_SRC  # noqa: E402
from peds_report_text import valid_volumes, texts_for, write_retrieval_npz  # noqa: E402
from open_clip import get_tokenizer, get_input_dtype  # noqa: E402
from open_clip_train.precision import get_autocast  # noqa: E402


def get_args():
    p = argparse.ArgumentParser('HLIP retrieval latents', parents=[zs_parser()], conflict_handler='resolve')
    p.add_argument('--out', default='/temp_work/ch278233/PEDS_BENCH/retrieval/hlip_valid.npz')
    p.add_argument('--text-batch', default=32, type=int)
    return p.parse_args()


def main(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    stems = valid_volumes()                                  # vollist order
    texts = texts_for(stems)
    print(f'[text] {len(stems)} volumes | empty texts: {sum(1 for t in texts if not t.strip())}', flush=True)
    for s, t in list(zip(stems, texts))[:2]:
        print(f'[text] {s}: {t[:60]!r}', flush=True)

    model, n_match = build_model(args)
    model.eval()
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    ctx = tokenizer.context_length
    print(f'[tokenizer] {type(tokenizer).__name__} {tokenizer.tokenizer.name_or_path} | context_length = {ctx}', flush=True)

    # truncation count at HLIP's own context length (special tokens included, as HFTokenizer adds them)
    raw_lens = [len(tokenizer.tokenizer(t, add_special_tokens=True, truncation=False)['input_ids']) for t in texts]
    n_trunc = sum(1 for n in raw_lens if n > ctx)
    print(f'[tokenizer] reports longer than {ctx} tokens (truncated): {n_trunc} / {len(texts)} | '
          f'token length median {int(np.median(raw_lens))} max {max(raw_lens)}', flush=True)

    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    # ---- image latents, identical loader to the zero-shot script, reordered to vollist order ----
    ds = ZeroShotDataset(args.data_root, args.split, args.labels, args.input_info)
    by_stem = {os.path.basename(p)[:-3]: (p, t) for p, t in ds.cts}
    missing = [s for s in stems if s not in by_stem]
    if missing:
        raise SystemExit(f'{len(missing)} vollist volumes have no preprocessed .pt / label row, e.g. {missing[:5]}')
    ds.cts = [by_stem[s] for s in stems]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    img_lat, order = [], []
    with torch.inference_mode():
        for recon, image, _ in tqdm(loader, total=len(loader), desc='image'):
            image = image.to(device=device, dtype=input_dtype)
            with autocast():
                out = model(image=image)
                f = out['image_features']
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            img_lat.append(f.cpu().numpy())
            order.append(os.path.basename(recon[0])[:-3])
    img_lat = np.concatenate(img_lat, 0).astype(np.float32)
    assert order == stems, 'image latent order does not match vollist order'

    # ---- text latents on the canonical report strings ----
    txt_lat = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), args.text_batch), desc='text'):
            tok = tokenizer(texts[i:i + args.text_batch]).to(device)
            with autocast():
                f = model.encode_text(tok, normalize=True)
            txt_lat.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    txt_lat = np.concatenate(txt_lat, 0).astype(np.float32)

    meta = {
        'model': args.model, 'checkpoint': args.resume, 'tensor_matches': int(n_match),
        'context_length': int(ctx), 'n_truncated': int(n_trunc), 'D': int(img_lat.shape[1]),
        'img_mean_l2': float(np.linalg.norm(img_lat, axis=1).mean()),
        'txt_mean_l2': float(np.linalg.norm(txt_lat, axis=1).mean()),
        'input_info': list(args.input_info), 'text_rule': 'peds_report_text.texts_for (impression first)',
    }
    path = write_retrieval_npz(args.out, img_lat, txt_lat, stems, meta=meta)
    print(f'[saved] {path}', flush=True)
    print('[meta] ' + json.dumps(meta), flush=True)


if __name__ == '__main__':
    main(get_args())
