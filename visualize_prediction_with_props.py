#!/usr/bin/env python3
"""
Create multi-panel visualizations for a prediction sample:
- Four property maps (E, nu, xi0, h0)
- Ground-truth von Mises (MPa)
- Predicted von Mises (MPa)
- Elastic estimate (Hooke) (MPa)
- Absolute prediction error map (MPa)

Input tensors are read from ML_DATASET, predictions from saved .npy files,
and grain-boundary overlays are derived from labels/seedXXXX.npy files.
"""
import argparse
import csv
import os
import random
from typing import Dict, List, Tuple

import numpy as np


def load_metadata(meta_csv: str) -> Tuple[Dict[str, dict], Dict[str, int]]:
    sample_info: Dict[str, dict] = {}
    max_inc_per_seed: Dict[str, int] = {}
    with open(meta_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = f"sample_{int(row['sample_idx']):05d}"
            info = {
                'seed': row['seed'],
                'split': row['split'],
                'increment_t': int(row['increment_t']),
            }
            sample_info[sample] = info
            seed = row['seed']
            inc = int(row['increment_t'])
            max_inc_per_seed[seed] = max(max_inc_per_seed.get(seed, -1), inc)
    return sample_info, max_inc_per_seed


def load_sample(data_root: str, split: str, sample: str) -> Tuple[np.ndarray, np.ndarray]:
    in_path = os.path.join(data_root, split, 'inputs', f'{sample}.npy')
    out_path = os.path.join(data_root, split, 'outputs', f'{sample}.npy')
    X = np.load(in_path).astype(np.float32)  # (H,W,C)
    Y = np.load(out_path).astype(np.float32)  # (H,W,1)
    return X, Y[..., 0]


def load_prediction(pred_dir: str, sample: str) -> np.ndarray:
    p = os.path.join(pred_dir, f'{sample}.npy')
    if not os.path.isfile(p):
        raise FileNotFoundError(f'Prediction not found: {p}')
    arr = np.load(p).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def denorm_props(X: np.ndarray, channels: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c0, c1, c2, c3 = channels
    E = X[..., c0] * 250e9 + 50e9
    nu = X[..., c1] * 0.2 + 0.2
    xi0 = X[..., c2] * 250e6 + 50e6
    h0 = X[..., c3] * 50e9
    return E, nu, xi0, h0


def gb_mask_from_labels(labels_path: str) -> np.ndarray:
    lbl = np.load(labels_path)
    if lbl.ndim != 2:
        raise ValueError(f'Labels must be 2D: {labels_path}')
    mask = np.zeros_like(lbl, dtype=bool)
    mask[:, :-1] |= (lbl[:, 1:] != lbl[:, :-1])
    mask[:-1, :] |= (lbl[1:, :] != lbl[:-1, :])
    return mask


def estimate_strain_fraction(increment_t: int, max_inc: int, initial_f11: float, final_f11: float) -> float:
    # target is increment t+1
    n_steps = max_inc + 1
    target_inc = increment_t + 1
    frac = target_inc / max(1, n_steps)
    F11 = initial_f11 + frac * (final_f11 - initial_f11)
    return F11 - 1.0


def plot_panels(sample: str,
                props: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                stress_gt_mpa: np.ndarray,
                stress_pred_mpa: np.ndarray,
                elastic_mpa: np.ndarray,
                abs_err_mpa: np.ndarray,
                gb_mask: np.ndarray | None,
                out_path: str):
    import matplotlib.pyplot as plt

    E, nu, xi0, h0 = props
    kmaps = [
        (E / 1e9, 'E (GPa)', 'viridis'),
        (nu, 'nu (-)', 'viridis'),
        (xi0 / 1e6, 'xi0 (MPa)', 'viridis'),
        (h0 / 1e9, 'h0 (GPa)', 'viridis'),
    ]
    smaps = [
        (stress_gt_mpa, 'σvM GT (MPa)', 'viridis'),
        (stress_pred_mpa, 'σvM Pred (MPa)', 'viridis'),
        (elastic_mpa, 'σvM elastic (MPa)', 'viridis'),
        (abs_err_mpa, '|Pred-GT| (MPa)', 'magma'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    for ax, (arr, title, cmap) in zip(axes.flatten(), kmaps + smaps):
        im = ax.imshow(arr, cmap=cmap, origin='lower', interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if gb_mask is not None:
            ax.contour(gb_mask.astype(float), levels=[0.5], colors='white', linewidths=0.6, alpha=0.7)

    fig.suptitle(sample, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ML_DATASET')
    ap.add_argument('--predictions', default=os.path.join('ML_EVAL', 'predictions'))
    ap.add_argument('--metadata', default=os.path.join('ML_DATASET', 'metadata', 'increments_map.csv'))
    ap.add_argument('--labels-dir', default='labels')
    ap.add_argument('--split', default='test')
    ap.add_argument('--samples', default='', help='comma-separated sample IDs (sample_XXXXX). Defaults to all in split.')
    ap.add_argument('--num', type=int, default=0, help='if >0 and --samples empty, randomly sample this many from split.')
    ap.add_argument('--prop-channels', default='0,1,2,3')
    ap.add_argument('--sigma-scale', type=float, default=1000.0)
    ap.add_argument('--initial-f11', type=float, default=1.0)
    ap.add_argument('--final-f11', type=float, default=1.004)
    ap.add_argument('--out', default=os.path.join('ML_EVAL', 'pred_prop_panels'))
    ap.add_argument('--seed', type=int, default=0, help='random seed for sampling')
    args = ap.parse_args()

    sample_info, max_inc_per_seed = load_metadata(args.metadata)
    prop_channels = tuple(int(x) for x in args.prop_channels.split(','))

    all_samples = [s for s, info in sample_info.items() if info['split'] == args.split]
    if args.samples.strip():
        selected = [s.strip() for s in args.samples.split(',') if s.strip()]
    elif args.num > 0 and args.num < len(all_samples):
        random.seed(args.seed or 0)
        selected = random.sample(all_samples, args.num)
    else:
        selected = sorted(all_samples)

    labels_cache: Dict[str, np.ndarray] = {}

    for sample in selected:
        info = sample_info.get(sample)
        if info is None:
            print(f"Skipping {sample}: not found in metadata.")
            continue
        split = info['split']
        seed = info['seed']
        increment_t = info['increment_t']
        max_inc = max_inc_per_seed[seed]

        X, Y = load_sample(args.data, split, sample)
        P_norm = load_prediction(args.predictions, sample)

        props = denorm_props(X, prop_channels)
        stress_gt_mpa = Y * args.sigma_scale
        stress_pred_mpa = P_norm * args.sigma_scale
        abs_err_mpa = np.abs(stress_pred_mpa - stress_gt_mpa)

        strain = estimate_strain_fraction(increment_t, max_inc, args.initial_f11, args.final_f11)
        E, nu, _, _ = props
        elastic_mpa = (E * strain / (1.0 + np.clip(nu, 1e-6, 0.49999))) / 1e6

        lbl_path = os.path.join(args.labels_dir, f'seed{seed}.npy')
        gb_mask = None
        if os.path.isfile(lbl_path):
            if seed in labels_cache:
                gb_mask = labels_cache[seed]
            else:
                gb_mask = gb_mask_from_labels(lbl_path)
                labels_cache[seed] = gb_mask
        else:
            print(f"Warning: missing labels for seed {seed}")

        out_path = os.path.join(args.out, f'{sample}_analysis.png')
        plot_panels(sample, props, stress_gt_mpa, stress_pred_mpa, elastic_mpa, abs_err_mpa, gb_mask, out_path)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()


