#!/usr/bin/env python3
"""
Generate a stress-field evolution video for a given seed:
- For each increment pair (t -> t+Δt) in the chosen split, plot GT, prediction, and abs error.
- Combine frames into an MP4 (or GIF fallback) and also save the final frame as a PNG.

Requires:
  * ML_DATASET inputs/outputs (normalized tensors)
  * Saved predictions (normalized) produced by evaluate_test.py --save-predictions
  * metadata CSV (increments_map.csv) to map samples → seed/increment
  * labels (optional) for grain-boundary overlays
"""
import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import imageio.v3 as iio


def load_metadata(meta_csv: str) -> Dict[str, Dict]:
    mapping: Dict[str, Dict] = {}
    with open(meta_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = f"sample_{int(row['sample_idx']):05d}"
            mapping[sample] = {
                'seed': row['seed'],
                'split': row['split'],
                'increment_t': int(row['increment_t']),
            }
    return mapping


def filter_samples(metadata: Dict[str, Dict], seed: str, split: str) -> List[Tuple[str, int]]:
    subset = [(sample, info['increment_t']) for sample, info in metadata.items()
              if info['seed'] == seed and info['split'] == split]
    subset.sort(key=lambda x: x[1])
    return subset


def load_sample_arrays(data_root: str, split: str, sample: str) -> Tuple[np.ndarray, np.ndarray]:
    y_path = os.path.join(data_root, split, 'outputs', f'{sample}.npy')
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing ground truth tensor: {y_path}")
    Y = np.load(y_path).astype(np.float32)[..., 0]
    return Y


def load_prediction(pred_dir: str, sample: str) -> np.ndarray:
    p_path = os.path.join(pred_dir, f'{sample}.npy')
    if not os.path.isfile(p_path):
        raise FileNotFoundError(f"Missing prediction tensor: {p_path}")
    arr = np.load(p_path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def gb_mask(labels_dir: str, seed: str) -> np.ndarray | None:
    lbl_path = os.path.join(labels_dir, f'seed{seed}.npy')
    if not os.path.isfile(lbl_path):
        alt = os.path.join(labels_dir, f'labels_seed{seed}.npy')
        if not os.path.isfile(alt):
            return None
        lbl_path = alt
    lbl = np.load(lbl_path)
    if lbl.ndim != 2:
        return None
    mask = np.zeros_like(lbl, dtype=bool)
    mask[:, :-1] |= (lbl[:, 1:] != lbl[:, :-1])
    mask[:-1, :] |= (lbl[1:, :] != lbl[:-1, :])
    return mask


def render_frame(sample: str,
                 gt_mpa: np.ndarray,
                 pred_mpa: np.ndarray,
                 err_mpa: np.ndarray) -> np.ndarray:
    vmin = float(min(gt_mpa.min(), pred_mpa.min()))
    vmax = float(max(gt_mpa.max(), pred_mpa.max()))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    entries = [
        (gt_mpa, 'Target (MPa)', 'viridis', vmin, vmax),
        (pred_mpa, 'Prediction (MPa)', 'viridis', vmin, vmax),
        (err_mpa, 'Abs Error (MPa)', 'magma', 0.0, float(err_mpa.max()))
    ]
    for ax, (arr, title, cmap, vmin_, vmax_) in zip(axes, entries):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin_, vmax=vmax_, origin='lower', interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(sample, fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return image


def save_video(frames: List[np.ndarray], out_path: str, fps: int = 4):
    ext = os.path.splitext(out_path)[1].lower()
    if ext == '.gif':
        iio.imwrite(out_path, frames, duration=int(1000 / fps))
    else:
        # default mp4
        iio.imwrite(out_path, frames, fps=fps, codec='libx264', quality=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ML_DATASET')
    ap.add_argument('--predictions', default=os.path.join('ML_EVAL', 'predictions'))
    ap.add_argument('--metadata', default=os.path.join('ML_DATASET', 'metadata', 'increments_map.csv'))
    ap.add_argument('--labels-dir', default='labels')
    ap.add_argument('--seed', required=True, help='Seed ID, e.g., 1099931136')
    ap.add_argument('--split', default='test')
    ap.add_argument('--sigma-scale', type=float, default=1000.0)
    ap.add_argument('--out-dir', default=os.path.join('ML_EVAL', 'stress_videos'))
    ap.add_argument('--video-name', default='')
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--format', default='mp4', choices=['mp4', 'gif'])
    args = ap.parse_args()

    metadata = load_metadata(args.metadata)
    samples = filter_samples(metadata, f'seed{args.seed}' if not args.seed.startswith('seed') else args.seed, args.split)
    if len(samples) == 0:
        raise RuntimeError(f"No samples found for seed={args.seed} split={args.split}")

    # metadata uses 'seed123456'; ensure consistent format
    first_sample = samples[0][0]
    seed_name = metadata[first_sample]['seed']
    sample_list = [s for s, _ in samples]

    gb = None  # overlays removed per request
    frames: List[np.ndarray] = []

    for sample in sample_list:
        info = metadata[sample]
        Y = load_sample_arrays(args.data, info['split'], sample)
        P = load_prediction(args.predictions, sample)
        gt_mpa = Y * args.sigma_scale
        pred_mpa = P * args.sigma_scale
        err_mpa = np.abs(pred_mpa - gt_mpa)
        frame = render_frame(sample, gt_mpa, pred_mpa, err_mpa)
        frames.append(frame)

    os.makedirs(args.out_dir, exist_ok=True)
    video_name = args.video_name or f"{seed_name}_{args.split}.{args.format}"
    video_path = os.path.join(args.out_dir, video_name)
    save_video(frames, video_path, fps=args.fps)
    final_frame_path = os.path.join(args.out_dir, f"{os.path.splitext(video_name)[0]}_final.png")
    iio.imwrite(final_frame_path, frames[-1])
    print(f"Saved video: {video_path}")
    print(f"Saved final frame: {final_frame_path}")


if __name__ == '__main__':
    main()

