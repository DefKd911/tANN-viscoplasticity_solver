#!/usr/bin/env python3
"""
Compare predicted vs ground-truth stress-strain curves per seed by reusing
saved per-sample predictions from evaluate_test.py.

The script assumes that:
1. evaluate_test.py was run with --save-predictions, producing
   <eval_out>/predictions/sample_XXXXX.npy files (normalized units).
2. ML_DATASET/metadata/increments_map.csv encodes which seed and increment
   each sample corresponds to (increment_t is the input increment; the target
   increment is increment_t+1).

Output: for each seed in the selected split (default: test) a CSV + plot
that reports strain (F11-1), mean von Mises stress from DAMASK (MPa),
and mean von Mises stress predicted by the surrogate.
"""
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def _load_metadata(metadata_csv: str, split: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split', '').strip() != split:
                continue
            rows.append({
                'sample_idx': int(row['sample_idx']),
                'seed': row['seed'].strip(),
                'increment_t': int(row['increment_t']),
            })
    return rows


def _sample_name(sample_idx: int) -> str:
    return f"sample_{sample_idx:05d}"


def _load_field(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Generate stress-strain curves per seed.")
    ap.add_argument('--data', default='ML_DATASET', help='Path to dataset root (expects split/outputs).')
    ap.add_argument('--split', default='test', help='Dataset split to analyze (train/val/test).')
    ap.add_argument('--predictions', required=True, help='Directory with saved predictions (*.npy).')
    ap.add_argument('--metadata', default=os.path.join('ML_DATASET', 'metadata', 'increments_map.csv'))
    ap.add_argument('--sigma-scale', type=float, default=1000.0, help='Factor mapping normalized stress -> MPa.')
    ap.add_argument('--initial-f11', type=float, default=1.0, help='Initial F11 value (usually 1.0).')
    ap.add_argument('--final-f11', type=float, default=1.004, help='Final F11 reached at the last increment.')
    ap.add_argument('--out', default=os.path.join('ML_EVAL', 'stress_strain'))
    ap.add_argument('--plot', action='store_true', help='Save stress-strain line plots per seed.')
    args = ap.parse_args()

    rows = _load_metadata(args.metadata, args.split)
    assert len(rows) > 0, f"No rows found for split={args.split} in {args.metadata}"

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row['seed']].append(row)

    os.makedirs(args.out, exist_ok=True)
    summary_rows = ['seed,num_points,true_slope,pred_slope,slope_error,mean_abs_error\n']

    try:
        import matplotlib.pyplot as plt  # optional
    except Exception:
        plt = None

    for seed, entries in grouped.items():
        entries.sort(key=lambda r: r['increment_t'])
        n_pairs = max(e['increment_t'] for e in entries) + 1  # number of (t->t+1) samples
        seed_rows = ['strain,F11,true_MPa,pred_MPa,abs_error\n']

        strains: List[float] = []
        true_vals: List[float] = []
        pred_vals: List[float] = []

        for row in entries:
            sample = _sample_name(row['sample_idx'])
            target_increment = row['increment_t'] + 1  # model predicts t+1
            frac = target_increment / float(n_pairs)
            F11 = args.initial_f11 + frac * (args.final_f11 - args.initial_f11)
            strain = F11 - 1.0

            true_path = os.path.join(args.data, args.split, 'outputs', f'{sample}.npy')
            pred_path = os.path.join(args.predictions, f'{sample}.npy')
            assert os.path.isfile(true_path), f"Missing ground-truth tensor: {true_path}"
            assert os.path.isfile(pred_path), f"Missing prediction tensor: {pred_path}"

            true_field = _load_field(true_path) * args.sigma_scale
            pred_field = _load_field(pred_path) * args.sigma_scale

            true_mean = float(true_field.mean())
            pred_mean = float(pred_field.mean())
            abs_err = abs(pred_mean - true_mean)

            strains.append(strain)
            true_vals.append(true_mean)
            pred_vals.append(pred_mean)

            seed_rows.append(f"{strain:.8e},{F11:.8f},{true_mean:.6f},{pred_mean:.6f},{abs_err:.6f}\n")

        # slope via linear fit (stress vs strain)
        true_slope = float(np.polyfit(strains, true_vals, 1)[0]) if len(strains) >= 2 else 0.0
        pred_slope = float(np.polyfit(strains, pred_vals, 1)[0]) if len(strains) >= 2 else 0.0
        slope_err = pred_slope - true_slope
        mean_abs = float(np.mean(np.abs(np.array(pred_vals) - np.array(true_vals))))
        summary_rows.append(f"{seed},{len(strains)},{true_slope:.3f},{pred_slope:.3f},{slope_err:.3f},{mean_abs:.3f}\n")

        seed_dir = os.path.join(args.out, 'csv')
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, f'{seed}_curve.csv'), 'w', encoding='utf-8') as f:
            for line in seed_rows:
                f.write(line)

        if args.plot and plt is not None:
            fig = plt.figure(figsize=(5,3), dpi=150)
            plt.plot(strains, true_vals, '-o', label='DAMASK', color='#1f77b4')
            plt.plot(strains, pred_vals, '-s', label='U-Net', color='#ff7f0e')
            plt.xlabel('Engineering strain (F11 - 1)')
            plt.ylabel('Mean σ_vM (MPa)')
            plt.title(f'Seed {seed}')
            plt.legend()
            plt.tight_layout()
            plot_dir = os.path.join(args.out, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'{seed}_curve.png'))
            plt.close(fig)

    with open(os.path.join(args.out, 'summary.csv'), 'w', encoding='utf-8') as f:
        for line in summary_rows:
            f.write(line)

    print(f"Written per-seed curves to {args.out}")


if __name__ == '__main__':
    main()


