"""
HLIP preprocessing for the BCH pediatric chest-CT benchmark.

This replicates, as exactly as possible, the transform in
    HLIP/data/ct_rate/process.py :: load_nifti_file / single_worker
namely:
    1. read the NIfTI with SimpleITK  -> array in (z, y, x) order
    2. apply the CT rescale (slope / intercept) so the array is in Hounsfield units
    3. resample to a target spacing of (3, 1, 1) mm in (z, y, x) with
       torch.nn.functional.interpolate(mode='trilinear') and
       target_size = (int(d*z_sp/3), int(h*y_sp/1), int(w*x_sp/1))
    4. keep the RAW HU values (windowing/normalisation happens at train/test time)
    5. save as a torch .pt tensor (float16 here, to save disk)

Differences from the CT-RATE script, and why:
  * CT-RATE stores the DICOM RescaleSlope/RescaleIntercept and the x/y/z spacing in a
    side-car metadata CSV, because its released NIfTI headers do not carry them.
    The BCH pediatric NIfTIs *do* carry them in the NIfTI header
    (scl_slope=1.0, scl_inter=-1024.0, pixdim=(x,y,z)), and SimpleITK applies the
    header slope/intercept on read.  So `sitk.ReadImage()` alone already yields HU,
    and the spacing is taken from `img.GetSpacing()` (which is (x, y, z)).
    `--rescale-slope/--rescale-intercept` default to the header values (auto); they can
    be overridden for auditing.
  * Output layout is flat (<out>/<split>/<volume>.pt) instead of CT-RATE's
    patient/study nesting, because the pediatric volume names are flat.
  * No re-orientation is performed -- process.py does not re-orient either.
"""

import os
import csv
import json
import time
import argparse
import traceback

import numpy as np
import SimpleITK as sitk
import torch


def get_args_parser():
    p = argparse.ArgumentParser('HLIP pediatric preprocessing', add_help=False)
    p.add_argument('--vollist', nargs='+',
                   default=['/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_train.txt',
                            '/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt'],
                   help='volume list file(s); split name is inferred from "_vollist_<split>.txt"')
    p.add_argument('--volume-map', default='/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv')
    p.add_argument('--save-dir', default='/temp_work/ch278233/PEDS_BENCH/hlip/pt/')
    p.add_argument('--info-dir', default='/temp_work/ch278233/PEDS_BENCH/hlip/info/')
    p.add_argument('--save-astype', default='float16', type=str, choices=['float16', 'float32'])
    p.add_argument('--spacing', nargs='+', default=[3, 1, 1], type=int, help='target (z, y, x) mm')
    p.add_argument('--rescale-slope', default=None, type=float,
                   help='override the NIfTI header slope (default: use header, applied by SimpleITK)')
    p.add_argument('--rescale-intercept', default=None, type=float,
                   help='override the NIfTI header intercept (default: use header, applied by SimpleITK)')
    p.add_argument('--shard', default=0, type=int)
    p.add_argument('--nshards', default=1, type=int)
    p.add_argument('--limit', default=0, type=int, help='debug: process at most N volumes')
    p.add_argument('--only', nargs='+', default=None, help='debug: only these volume names')
    p.add_argument('--overwrite', default=False, action='store_true')
    p.add_argument('--max-total-gb', default=0.0, type=float,
                   help='hard budget for the whole save-dir (GB, decimal). 0 disables. '
                        'Every shard checks the shared total and aborts rather than exceeding it.')
    p.add_argument('--cap-check-every', default=10, type=int,
                   help='re-measure the save-dir size every N saved volumes')
    p.add_argument('--dry-run', default=False, action='store_true', help='report shapes, write nothing')
    return p


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def split_of(vollist_path):
    base = os.path.basename(vollist_path)
    stem = os.path.splitext(base)[0]
    return stem.rsplit('_', 1)[-1]


def read_volume_map(path):
    mapping = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            mapping[row['volume_name']] = row['nii_path']
    return mapping


def build_tasks(args):
    mapping = read_volume_map(args.volume_map)
    tasks = []
    for vl in args.vollist:
        split = split_of(vl)
        with open(vl) as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                if args.only is not None and name not in args.only:
                    continue
                stem = name.split('.')[0]              # ped_00002_1.nii.gz -> ped_00002_1
                nii = mapping.get(name)
                tasks.append((split, stem, name, nii))
    tasks.sort(key=lambda t: (t[0], t[1]))
    if args.nshards > 1:
        tasks = tasks[args.shard::args.nshards]
    if args.limit:
        tasks = tasks[:args.limit]
    return tasks


def load_and_resample(nii_path, args, verbose=False):
    """Returns (tensor, meta). Mirrors process.py::load_nifti_file."""
    img = sitk.ReadImage(nii_path)

    # spacing, SimpleITK returns (x, y, z)
    x, y, z = map(float, img.GetSpacing())

    # optional explicit rescale override (SimpleITK already applied the header slope/intercept)
    if args.rescale_slope is not None or args.rescale_intercept is not None:
        slope = 1.0 if args.rescale_slope is None else args.rescale_slope
        inter = 0.0 if args.rescale_intercept is None else args.rescale_intercept
        img = img * slope + inter

    # array: d, h, w  (z, y, x)
    arr = sitk.GetArrayFromImage(img)

    in_shape = tuple(arr.shape)
    in_min, in_max = float(arr.min()), float(arr.max())

    # --- header sanity flags (reported only; the transform itself stays faithful to process.py) ---
    direction = img.GetDirection()                       # row-major 3x3, columns = image axes in LPS
    axis3 = (direction[2], direction[5], direction[8])   # direction of the 3rd (slice) image axis
    z_fov = in_shape[0] * z
    anomalies = []
    if abs(axis3[2]) < 0.9:
        anomalies.append(f'non-axial slice axis: 3rd image axis points {tuple(round(a, 3) for a in axis3)} in LPS')
    if not (30.0 <= z_fov <= 600.0):
        anomalies.append(f'implausible z extent: {in_shape[0]} slices x {z:.4f} mm = {z_fov:.1f} mm')

    target_size = (
        int(arr.shape[0] * z / args.spacing[0]),
        int(arr.shape[1] * y / args.spacing[1]),
        int(arr.shape[2] * x / args.spacing[2]),
    )
    if min(target_size) < 1:
        raise ValueError(f'degenerate target size {target_size} for shape {in_shape} spacing {(z, y, x)}')

    out = torch.nn.functional.interpolate(
        torch.from_numpy(arr).float()[None, None, ...], size=target_size, mode='trilinear'
    ).squeeze()

    meta = {
        'in_shape_zyx': in_shape,
        'in_spacing_zyx': (z, y, x),
        'in_hu_min': in_min, 'in_hu_max': in_max,
        'out_shape_zyx': tuple(out.shape),
        'out_spacing_zyx': tuple(args.spacing),
        'out_hu_min': float(out.min()), 'out_hu_max': float(out.max()),
        'anomalies': anomalies,
    }
    if anomalies:
        print('  [ANOMALY] ' + ' ; '.join(anomalies), flush=True)
    if verbose:
        print(f'  in  shape (z,y,x) = {in_shape}  spacing (z,y,x) = ({z}, {y}, {x})  HU [{in_min}, {in_max}]')
        print(f'  expected out      = ({in_shape[0]}*{z}/{args.spacing[0]}, {in_shape[1]}*{y}/{args.spacing[1]}, '
              f'{in_shape[2]}*{x}/{args.spacing[2]}) = '
              f'({int(in_shape[0]*z/args.spacing[0])}, {int(in_shape[1]*y/args.spacing[1])}, {int(in_shape[2]*x/args.spacing[2])})')
        print(f'  out shape (z,y,x) = {tuple(out.shape)}  HU [{meta["out_hu_min"]:.1f}, {meta["out_hu_max"]:.1f}]')
    return out, meta


def main(args):
    tasks = build_tasks(args)
    print(f'[shard {args.shard}/{args.nshards}] {len(tasks)} volumes', flush=True)
    results = {}
    t_start = time.time()
    cap_bytes = int(args.max_total_gb * 1e9) if args.max_total_gb else 0
    n_saved = 0
    if cap_bytes:
        used = dir_bytes(args.save_dir)
        print(f'[budget] cap {args.max_total_gb:.1f} GB | save-dir currently {used/1e9:.2f} GB', flush=True)
        if used >= cap_bytes:
            print(f'[STOP] budget already exhausted before starting ({used/1e9:.2f} GB >= {args.max_total_gb:.1f} GB)', flush=True)
            raise SystemExit(2)
    for i, (split, stem, name, nii) in enumerate(tasks):
        if cap_bytes and n_saved and n_saved % args.cap_check_every == 0:
            used = dir_bytes(args.save_dir)
            if used >= cap_bytes:
                print(f'[STOP] budget reached: save-dir {used/1e9:.2f} GB >= cap {args.max_total_gb:.1f} GB '
                      f'after {n_saved} volumes in this shard. Stopping rather than exceeding it.', flush=True)
                results['__stopped_on_budget__'] = f'{used/1e9:.2f} GB'
                break
        out_path = os.path.join(args.save_dir, split, stem + '.pt')
        if (not args.overwrite) and os.path.exists(out_path) and not args.dry_run:
            results[stem] = 'skip (exists)'
            continue
        if nii is None or not os.path.exists(nii):
            print(f'[FAIL] {stem}: missing nii ({nii})', flush=True)
            results[stem] = 'fail: missing nii'
            continue
        t0 = time.time()
        try:
            out, meta = load_and_resample(nii, args, verbose=args.dry_run)
        except Exception:
            print(f'[FAIL] {stem}\n{traceback.format_exc()}', flush=True)
            results[stem] = 'fail: ' + traceback.format_exc().strip().splitlines()[-1]
            continue

        out = out.to(torch.float16) if args.save_astype == 'float16' else out.to(torch.float32)
        if not args.dry_run:
            # the save is retried: the shared filesystem occasionally stalls, and a failed
            # write must not kill the whole shard (a later re-run resumes on missing files)
            saved = False
            for attempt in range(3):
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    tmp = out_path + f'.tmp{os.getpid()}'
                    torch.save(out, tmp)
                    os.replace(tmp, out_path)
                    saved = True
                    break
                except Exception:
                    print(f'[WARN] save attempt {attempt+1} failed for {stem}\n{traceback.format_exc()}', flush=True)
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass
                    time.sleep(10)
            if not saved:
                print(f'[FAIL] {stem}: could not write {out_path}', flush=True)
                results[stem] = 'fail: save failed'
                continue
            n_saved += 1
        results[stem] = meta
        print(f'[{i+1}/{len(tasks)}] {split}/{stem} {meta["in_shape_zyx"]} sp{tuple(round(s,4) for s in meta["in_spacing_zyx"])}'
              f' -> {meta["out_shape_zyx"]} HU[{meta["out_hu_min"]:.0f},{meta["out_hu_max"]:.0f}] {time.time()-t0:.1f}s', flush=True)

    if not args.dry_run:
        os.makedirs(args.info_dir, exist_ok=True)
        with open(os.path.join(args.info_dir, f'info_shard{args.shard:04d}_of{args.nshards:04d}.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    n_fail = sum(1 for v in results.values() if isinstance(v, str) and v.startswith('fail'))
    print(f'[shard {args.shard}] done in {time.time()-t_start:.1f}s, {len(results)} handled, {n_fail} failed', flush=True)
    if '__stopped_on_budget__' in results:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HLIP pediatric preprocessing', parents=[get_args_parser()])
    main(parser.parse_args())
