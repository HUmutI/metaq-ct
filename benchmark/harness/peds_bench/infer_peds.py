"""fVLM zero-shot inference on the BCH pediatric validation cohort.

Adapted from fvlm/eval.py. CHANGES vs the repo (no architecture change):
  * eval.py hardcodes `multi-modal-results/pretrain_ckpts/xxx/checkpoint_{epoch}.pth`
    and loops epoch in range(10,20); we load the single released CT-RATE model.pth.
  * data loading reads our preprocessed .npz instead of walking data/processed_valid_images.
  * single-GPU (no torchrun / distributed).
The scoring logic (center_crop -> DivisiblePad -> forward_test_win -> mean of
softmax probs, positive index 1) is copied verbatim in structure from eval.py.

CLASS VOCABULARY -- IMPORTANT:
  fVLM is zero-shot via per-(organ, abnormality) text prototypes. Its released
  evaluation defines 16 (organ, abnormality) test items over the CT-RATE label set.
  12 of our 23 pediatric classes appear verbatim in that list ("official" prompts,
  reproduced character-for-character from eval.py). The other 11 pediatric classes
  have NO fVLM/CT-RATE prototype; we build prompts for them with the *same template*
  ("Not X." / "X.") and assign the anatomically closest of fVLM's 4 organs.
  These 11 are EXTRAPOLATED, flagged in the output as class_is_official=False, and
  must be reported separately. The headline "no pediatric training" number should be
  the 12 official-vocabulary classes.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from typing import Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PEDS23 = ['Medical material', 'Cardiomegaly', 'Pericardial effusion', 'Lymphadenopathy',
          'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
          'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening',
          'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening',
          'Post-surgical or post-treatment change', 'Pulmonary metastases', 'Tree-in-bud',
          'Pulmonary cyst', 'Mass or neoplasm', 'Mucus plugging',
          'Pleural thickening or nodule', 'Bone lesion or fracture', 'Pneumothorax']

# --- verbatim from fvlm/eval.py DataFolder.test_items ---------------------------
OFFICIAL_ITEMS = {
    'Emphysema':                       ['lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'],
    'Atelectasis':                     ['lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'],
    'Lung nodule':                     ['lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'],
    'Lung opacity':                    ['lung', 'Lung opacity', 'Not Opacity.', 'Opacity.'],
    'Pulmonary fibrotic sequela':      ['lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'],
    'Pleural effusion':                ['lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'],
    'Mosaic attenuation pattern':      ['lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'],
    'Peribronchial thickening':        ['lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'],
    'Consolidation':                   ['lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'],
    'Bronchiectasis':                  ['lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'],
    'Interlobular septal thickening':  ['lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'],
    'Cardiomegaly':                    ['heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'],
    'Pericardial effusion':            ['heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'],
    'Coronary artery wall calcification': ['heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.', 'Coronary artery wall calcification.'],
    'Hiatal hernia':                   ['esophagus', 'Hiatal hernia', 'Not Hiatal hernia.', 'Hiatal hernia.'],
    'Arterial wall calcification':     ['aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'],
}
# --- extrapolated: same template, closest fVLM organ (NOT in fVLM's vocabulary) ---
EXTRA_ITEMS = {
    'Medical material':                       ['lung',  'Not Medical material.', 'Medical material.'],
    'Lymphadenopathy':                        ['heart', 'Not Lymphadenopathy.', 'Lymphadenopathy.'],
    'Post-surgical or post-treatment change': ['lung',  'Not Post-surgical or post-treatment change.', 'Post-surgical or post-treatment change.'],
    'Pulmonary metastases':                   ['lung',  'Not Pulmonary metastases.', 'Pulmonary metastases.'],
    'Tree-in-bud':                            ['lung',  'Not Tree-in-bud.', 'Tree-in-bud.'],
    'Pulmonary cyst':                         ['lung',  'Not Pulmonary cyst.', 'Pulmonary cyst.'],
    'Mass or neoplasm':                       ['lung',  'Not Mass or neoplasm.', 'Mass or neoplasm.'],
    'Mucus plugging':                         ['lung',  'Not Mucus plugging.', 'Mucus plugging.'],
    'Pleural thickening or nodule':           ['lung',  'Not Pleural thickening or nodule.', 'Pleural thickening or nodule.'],
    'Bone lesion or fracture':                ['lung',  'Not Bone lesion or fracture.', 'Bone lesion or fracture.'],
    'Pneumothorax':                           ['lung',  'Not Pneumothorax.', 'Pneumothorax.'],
}

def build_test_items():
    items, official = [], []
    for c in PEDS23:
        if c in OFFICIAL_ITEMS:
            items.append(tuple(OFFICIAL_ITEMS[c])); official.append(True)
        else:
            o, neg, pos = EXTRA_ITEMS[c]
            items.append((o, c, neg, pos)); official.append(False)
    return items, np.array(official)

# ---- verbatim helpers from fvlm/eval.py ---------------------------------------
def masks_to_boxes_3d(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)
    d, h, w = masks.shape[-3:]
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')
    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)

def center_crop(image, mask, crop_size):
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    crop_d, crop_h, crop_w = (max(crop_size[0], z_max - z_min),
                              max(crop_size[1], y_max - y_min),
                              max(crop_size[2], x_max - x_min))
    cx = (x_min + x_max) // 2; cy = (y_min + y_max) // 2; cz = (z_min + z_max) // 2
    d, h, w = image.shape[-3:]
    x_start = max(0, cx - crop_w // 2); x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w: x_start = max(0, x_end - crop_w)
    y_start = max(0, cy - crop_h // 2); y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h: y_start = max(0, y_end - crop_h)
    z_start = max(0, cz - crop_d // 2); z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d: z_start = max(0, z_end - crop_d)
    return (image[..., z_start:z_end, y_start:y_end, x_start:x_end],
            mask[...,  z_start:z_end, y_start:y_end, x_start:x_end])

ORGANS = ['lung', 'heart', 'esophagus', 'aorta']
ROI_SIZE = (112, 288, 352)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vollist', default='/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--qc-out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()

    from monai import transforms
    from torch.utils.data import DataLoader
    from fvlm_model import build_model
    from peds_data import PedsVolumes, load_vol_map, collate, ORGANS

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt_info = build_model(device=dev)
    print('CKPT_INFO ' + json.dumps({k: ckpt_info[k] for k in ('n_ckpt','n_model','matched')}), flush=True)

    test_items, is_official = build_test_items()
    text_feat_dict = model.prepare_text_feat(test_items)
    print(f'text prototypes: {len(text_feat_dict)} (official={int(is_official.sum())} extrapolated={int((~is_official).sum())})', flush=True)

    pad_func = transforms.DivisiblePadd(keys=["image", "label"], k=(16, 16, 32),
                                        mode='constant', constant_values=0, method="end")

    vols = [l.strip() for l in open(args.vollist) if l.strip()]
    if args.limit: vols = vols[:args.limit]
    vols = vols[args.shard::args.nshards]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try: done.add(json.loads(line)['volume'])
            except Exception: pass
    vols = [v for v in vols if v not in done]
    print(f'shard {args.shard}: {len(vols)} volumes to score', flush=True)

    ds = PedsVolumes(vols, load_vol_map())
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                    collate_fn=collate, prefetch_factor=2 if args.workers else None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.qc_out), exist_ok=True)
    fout = open(args.out, 'a'); fqc = open(args.qc_out, 'a')

    t0 = time.time()
    for n, (image, mask, qc) in enumerate(dl):
        fqc.write(json.dumps(qc) + '\n'); fqc.flush()
        vol = qc['volume']
        if image is None:
            fout.write(json.dumps({'volume': vol, 'status': qc['status']}) + '\n'); fout.flush()
            continue
        try:
            image = image[None].to(dev).float()
            mask = mask[None].to(dev).float()

            whole_organ_sizes = {o: torch.eq(mask, ORGANS.index(o) + 1).sum().item() for o in ORGANS}
            test_organs = [o for o in ORGANS if whole_organ_sizes[o] > 0]
            items = [it for it in test_items if it[0] in test_organs]

            organ_logits = {it: [] for it in items}
            organ_feat = {}
            for k, v in organ_logits.items():
                if not len(v):
                    organ_id = ORGANS.index(k[0])
                    wp, wm = center_crop(image, torch.eq(mask, organ_id + 1), crop_size=ROI_SIZE)
                    wm = wm.float(); wm[wm == 1] = organ_id + 1
                    pd = pad_func({'image': wp[0], 'label': wm[0]})
                    wp, wm = pd['image'], pd['label']
                    organ_logits = model.forward_test_win(
                        wp[None], wm[None], organ_logits, test_organs,
                        text_feat_dict, organ_feat, whole_organ_sizes, skip_organ=organ_id)

            rec = {'volume': vol, 'status': 'OK', 'organs_present': test_organs,
                   'probs': {}, 'nwin': {}}
            for it, probs in organ_logits.items():
                if len(probs):
                    arr = np.concatenate(probs)
                    rec['probs'][it[1]] = float(arr.mean(0)[1])
                    rec['nwin'][it[1]] = int(arr.shape[0])
            fout.write(json.dumps(rec) + '\n'); fout.flush()
            del image, mask
        except Exception as e:
            import traceback; traceback.print_exc()
            fout.write(json.dumps({'volume': vol, 'status': f'FAIL:{type(e).__name__}:{e}'}) + '\n'); fout.flush()
        if n % 10 == 0:
            el = time.time() - t0
            print(f'[{args.shard}] {n}/{len(vols)} {el:.0f}s ({el/max(1,n+1):.2f}s/vol)', flush=True)
    fout.close(); fqc.close()
    print('SHARD_DONE', args.shard, flush=True)

if __name__ == '__main__':
    main()
