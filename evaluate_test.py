#!/usr/bin/env python3
"""
Evaluate U-Net on ML_DATASET/test and save metrics and qualitative plots.

Usage:
  python evaluate_test.py --data ML_DATASET --ckpt ML_CHECKPOINTS/best.pt --out ML_EVAL --batch-size 16 --base 32
"""
import os
import glob
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import csv
import re


class NpyPairDataset(Dataset):
    def __init__(self, root_split_dir: str):
        self.in_paths = sorted(glob.glob(os.path.join(root_split_dir, 'inputs', 'sample_*.npy')))
        self.out_paths = [p.replace(os.path.sep + 'inputs' + os.path.sep, os.path.sep + 'outputs' + os.path.sep) for p in self.in_paths]
        assert len(self.in_paths) == len(self.out_paths) and len(self.in_paths) > 0, f"No samples in {root_split_dir}"
    def __len__(self):
        return len(self.in_paths)
    def __getitem__(self, idx):
        X = np.load(self.in_paths[idx]).astype(np.float32)
        Y = np.load(self.out_paths[idx]).astype(np.float32)
        X = torch.from_numpy(X).permute(2,0,1)
        Y = torch.from_numpy(Y).permute(2,0,1)
        return X, Y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k//2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = DoubleConv(base*8 + base*4, base*4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = DoubleConv(base*4 + base*2, base*2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = DoubleConv(base*2 + base, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bn = self.bottleneck(self.pool3(d3))
        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        out = self.outc(u1)
        return out


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    # Match training: compute per-sample mean losses and average across samples
    l1 = nn.L1Loss(reduction='mean')
    l2 = nn.MSELoss(reduction='mean')
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        P = model(X)
        bsz = X.size(0)
        mae_sum += float(l1(P, Y).item()) * bsz
        mse_sum += float(l2(P, Y).item()) * bsz
        n += bsz
    mae = mae_sum / max(1, n)
    rmse = float(np.sqrt(mse_sum / max(1, n)))
    return mae, rmse


@torch.no_grad()
def _compute_boundary_mask_from_channel(ch_map: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Return a boolean mask (H,W) where 4-neighborhood changes indicate grain boundaries.
    ch_map is a single-channel 2D array already on CPU (normalized)."""
    H, W = ch_map.shape
    mask = np.zeros((H, W), dtype=bool)
    # Compare with right and down neighbors
    mask[:, :-1] |= np.abs(ch_map[:, 1:] - ch_map[:, :-1]) > tol
    mask[:-1, :] |= np.abs(ch_map[1:, :] - ch_map[:-1, :]) > tol
    return mask


def _compute_boundary_mask_from_labels(lbl: np.ndarray) -> np.ndarray:
    H, W = lbl.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:, :-1] |= (lbl[:, 1:] != lbl[:, :-1])
    mask[:-1, :] |= (lbl[1:, :] != lbl[:-1, :])
    return mask


def _edges_from_scalar_field(arr: np.ndarray, tol: float) -> np.ndarray:
    H, W = arr.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:, :-1] |= np.abs(arr[:, 1:] - arr[:, :-1]) > tol
    mask[:-1, :] |= np.abs(arr[1:, :] - arr[:-1, :]) > tol
    return mask


def _apply_transform(arr: np.ndarray, t: str) -> np.ndarray:
    if t == 'identity':
        return arr
    if t == 'rot90':
        return np.rot90(arr, 1)
    if t == 'rot180':
        return np.rot90(arr, 2)
    if t == 'rot270':
        return np.rot90(arr, 3)
    if t == 'flipud':
        return np.flipud(arr)
    if t == 'fliplr':
        return np.fliplr(arr)
    if t == 'rot90_fliplr':
        return np.fliplr(np.rot90(arr, 1))
    if t == 'rot90_flipud':
        return np.flipud(np.rot90(arr, 1))
    return arr


def _auto_align_field_to_labels(field: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, str]:
    """Align scalar field to labels by maximizing edge overlap."""
    lbl_edges = _compute_boundary_mask_from_labels(labels)
    field_range = float(field.max() - field.min())
    tol = max(1e-6, 0.05 * field_range)
    transforms = ['identity', 'rot90', 'rot180', 'rot270', 'flipud', 'fliplr', 'rot90_fliplr', 'rot90_flipud']
    best_score = -1.0
    best_t = 'identity'
    for t in transforms:
        ft = _apply_transform(field, t)
        if ft.shape != labels.shape:
            continue
        field_edges = _edges_from_scalar_field(ft, tol=tol)
        inter = np.logical_and(lbl_edges, field_edges).sum()
        union = np.logical_or(lbl_edges, field_edges).sum()
        score = inter / max(1, union)
        if score > best_score:
            best_score = score
            best_t = t
            best_field = ft
    return best_field if best_score >= 0 else field, best_t


def _find_labels_path(labels_dir: str, seed: str) -> str | None:
    candidates = [
        os.path.join(labels_dir, f'seed{seed}.npy'),
        os.path.join(labels_dir, f'labels_seed{seed}.npy'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _load_sample_to_seed_map(metadata_csv_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.isfile(metadata_csv_path):
        return mapping
    try:
        with open(metadata_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_name: Optional[str] = None
                seed_str: Optional[str] = None
                if 'sample' in row and row['sample']:
                    sample_name = os.path.basename(row['sample'].strip()).replace('.npy','')
                elif 'sample_idx' in row and row['sample_idx']:
                    try:
                        idx = int(row['sample_idx'])
                        sample_name = f"sample_{idx:05d}"
                    except Exception:
                        sample_name = None
                if 'seed' in row and row['seed']:
                    seed_str = re.sub(r'\D', '', row['seed'])
                if not seed_str:
                    for val in row.values():
                        m = re.search(r'seed(\d+)', str(val))
                        if m:
                            seed_str = m.group(1)
                            break
                if sample_name and seed_str:
                    mapping[sample_name] = seed_str
    except Exception:
        return {}
    return mapping


@torch.no_grad()
def save_qualitative(model, ds: NpyPairDataset, out_dir: str, device, num_samples: int = 3, overlay_boundaries: bool = False, boundary_channel: int = 0, sigma_scale: float = 1000.0, export_mpa_plots: bool = False, sample_to_seed: Dict[str, str] | None = None, labels_dir: str | None = None, auto_align_boundaries: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:max(1, num_samples)]
    seed_alignment_cache: Dict[str, Tuple[str, np.ndarray]] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: matplotlib not available for qualitative plots: {e}")
        return
    for idx in idxs:
        X, Y = ds[idx]
        Xb = X.unsqueeze(0).to(device)
        Pb = model(Xb)
        P = Pb.squeeze(0).detach().cpu().numpy()  # (1,H,W)
        T = Y.numpy()                     # (1,H,W)
        P2 = P[0]
        T2 = T[0]
        E = np.abs(P2 - T2)

        transform = 'identity'
        bmask = None
        seed = None
        if overlay_boundaries:
            # Prefer exact labels-based boundaries if available
            name = os.path.basename(ds.in_paths[idx]).replace('.npy','')
            if sample_to_seed is not None and labels_dir is not None:
                seed = sample_to_seed.get(name)
                if seed is not None:
                    cache_entry = seed_alignment_cache.get(seed)
                    if cache_entry is not None:
                        transform, bmask_cached = cache_entry
                        if transform != 'identity':
                            T2 = _apply_transform(T2, transform)
                            P2 = _apply_transform(P2, transform)
                            E = _apply_transform(E, transform)
                        bmask = bmask_cached
                    else:
                        lbl_path = _find_labels_path(labels_dir, seed)
                        if lbl_path and os.path.isfile(lbl_path):
                            try:
                                lbl = np.load(lbl_path)
                                if lbl.ndim == 2:
                                    if auto_align_boundaries:
                                        aligned_field, transform = _auto_align_field_to_labels(T2, lbl)
                                        if transform != 'identity':
                                            T2 = aligned_field
                                            P2 = _apply_transform(P2, transform)
                                            E = _apply_transform(E, transform)
                                    bmask = _compute_boundary_mask_from_labels(lbl)
                                    seed_alignment_cache[seed] = (transform, bmask)
                                else:
                                    seed_alignment_cache[seed] = ('identity', None)
                            except Exception:
                                seed_alignment_cache[seed] = ('identity', None)
                        else:
                            seed_alignment_cache[seed] = ('identity', None)
                        cache_entry = seed_alignment_cache.get(seed)
                        transform = cache_entry[0]
                        bmask = cache_entry[1]
            if bmask is None:
                # fallback to boundary derived from selected input channel
                ch = X[boundary_channel].numpy()
                if transform != 'identity':
                    ch = _apply_transform(ch, transform)
                bmask = _compute_boundary_mask_from_channel(ch)
        vmin = float(min(P2.min(), T2.min()))
        vmax = float(max(P2.max(), T2.max()))

        fig = plt.figure(figsize=(9,3), dpi=150)
        ax1 = plt.subplot(1,3,1)
        im1 = ax1.imshow(T2, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
        ax1.set_title('Target (normalized)')
        ax1.axis('off')
        plt.colorbar(im1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(1,3,2)
        im2 = ax2.imshow(P2, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
        ax2.set_title('Prediction (normalized)')
        ax2.axis('off')
        plt.colorbar(im2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(1,3,3)
        im3 = ax3.imshow(E, cmap='magma', origin='lower', interpolation='nearest')
        ax3.set_title('Abs Error')
        ax3.axis('off')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        if overlay_boundaries and bmask is not None:
            for ax in (ax1, ax2, ax3):
                ax.contour(bmask.astype(float), levels=[0.5], colors='white', linewidths=0.5, alpha=0.7, origin='lower')

        plt.tight_layout()
        name = os.path.basename(ds.in_paths[idx]).replace('.npy','')
        plt.savefig(os.path.join(out_dir, f'{name}.png'))
        plt.close(fig)

        # Optional MPa plots
        if export_mpa_plots:
            T_mpa = T2 * sigma_scale
            P_mpa = P2 * sigma_scale
            E_mpa = E * sigma_scale
            vmin_mpa = float(min(T_mpa.min(), P_mpa.min()))
            vmax_mpa = float(max(T_mpa.max(), P_mpa.max()))

            fig = plt.figure(figsize=(9,3), dpi=150)
            ax1 = plt.subplot(1,3,1)
            im1 = ax1.imshow(T_mpa, cmap='viridis', vmin=vmin_mpa, vmax=vmax_mpa, origin='lower', interpolation='nearest')
            ax1.set_title('Target (MPa)')
            ax1.axis('off')
            plt.colorbar(im1, fraction=0.046, pad=0.04)

            ax2 = plt.subplot(1,3,2)
            im2 = ax2.imshow(P_mpa, cmap='viridis', vmin=vmin_mpa, vmax=vmax_mpa, origin='lower', interpolation='nearest')
            ax2.set_title('Prediction (MPa)')
            ax2.axis('off')
            plt.colorbar(im2, fraction=0.046, pad=0.04)

            ax3 = plt.subplot(1,3,3)
            im3 = ax3.imshow(E_mpa, cmap='magma', origin='lower', interpolation='nearest')
            ax3.set_title('Abs Error (MPa)')
            ax3.axis('off')
            plt.colorbar(im3, fraction=0.046, pad=0.04)

            if overlay_boundaries:
                if bmask is None:
                    ch = X[boundary_channel].numpy()
                    if transform != 'identity':
                        ch = _apply_transform(ch, transform)
                    bmask = _compute_boundary_mask_from_channel(ch)
                for ax in (ax1, ax2, ax3):
                    ax.contour(bmask.astype(float), levels=[0.5], colors='white', linewidths=0.5, alpha=0.7, origin='lower')

            plt.tight_layout()
            qdir = os.path.join(out_dir, 'mpa')
            os.makedirs(qdir, exist_ok=True)
            plt.savefig(os.path.join(qdir, f'{name}_mpa.png'))
            plt.close(fig)


def _compute_error_stats(abs_err: np.ndarray):
    # abs_err shape: (H,W)
    flat = abs_err.reshape(-1)
    mae = float(np.mean(flat))
    rmse = float(np.sqrt(np.mean(flat ** 2)))
    p50, p90, p95, p99, p100 = np.percentile(flat, [50, 90, 95, 99, 100]).tolist()
    return mae, rmse, p50, p90, p95, p99, p100


@torch.no_grad()
def save_per_sample_stats_and_hist(
    model,
    ds: NpyPairDataset,
    out_dir: str,
    device,
    bins: int = 50,
    hist_samples: int = 12,
    save_all_hists: bool = False,
    sigma_scale: float = 1000.0,
    save_predictions: bool = False,
    predictions_dir: str | None = None,
    save_mean_table: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    hist_dir = os.path.join(out_dir, 'histograms')
    if save_all_hists or hist_samples > 0:
        os.makedirs(hist_dir, exist_ok=True)

    if save_predictions:
        if predictions_dir is None:
            predictions_dir = os.path.join(out_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'per_sample_errors.csv')
    csv_path_mpa = os.path.join(out_dir, 'per_sample_errors_mpa.csv')
    rows: List[str] = []
    rows.append('sample,mae,rmse,p50,p90,p95,p99,max\n')
    rows_mpa: List[str] = []
    rows_mpa.append('sample,mae_MPa,rmse_MPa,p50_MPa,p90_MPa,p95_MPa,p99_MPa,max_MPa\n')
    mean_rows: List[str] = []
    if save_mean_table:
        mean_rows.append('sample,mean_true,mean_pred,mean_true_MPa,mean_pred_MPa\n')

    # choose which samples to save histograms for
    idx_list = list(range(len(ds)))
    rand_idxs = set(random.sample(idx_list, k=min(hist_samples, len(ds)))) if hist_samples > 0 else set()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    all_flat_errors: List[np.ndarray] = []

    for idx in idx_list:
        X, Y = ds[idx]
        Xb = X.unsqueeze(0).to(device)
        Pb = model(Xb)
        P = Pb.squeeze(0).cpu().numpy()  # (1,H,W)
        T = Y.numpy()                     # (1,H,W)
        E = np.abs(P[0] - T[0])           # (H,W)

        if save_predictions:
            name_pred = os.path.basename(ds.in_paths[idx]).replace('.npy','')
            pred_path = os.path.join(predictions_dir, f'{name_pred}.npy')
            np.save(pred_path, P.astype(np.float32))

        mae, rmse, p50, p90, p95, p99, p100 = _compute_error_stats(E)
        name = os.path.basename(ds.in_paths[idx]).replace('.npy','')
        rows.append(f"{name},{mae:.8f},{rmse:.8f},{p50:.8f},{p90:.8f},{p95:.8f},{p99:.8f},{p100:.8f}\n")
        rows_mpa.append(
            f"{name},{mae*sigma_scale:.8f},{rmse*sigma_scale:.8f},{p50*sigma_scale:.8f},{p90*sigma_scale:.8f},{p95*sigma_scale:.8f},{p99*sigma_scale:.8f},{p100*sigma_scale:.8f}\n"
        )

        if save_mean_table:
            mean_true = float(T[0].mean())
            mean_pred = float(P[0].mean())
            mean_rows.append(
                f"{name},{mean_true:.8f},{mean_pred:.8f},{mean_true*sigma_scale:.8f},{mean_pred*sigma_scale:.8f}\n"
            )

        all_flat_errors.append(E.reshape(-1))

        if plt is not None and (save_all_hists or idx in rand_idxs):
            fig = plt.figure(figsize=(4,3), dpi=150)
            plt.hist(E.reshape(-1), bins=bins, color='#4C78A8')
            plt.xlabel('Absolute error (normalized)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f'{name}_hist.png'))
            plt.close(fig)

    with open(csv_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(r)
    with open(csv_path_mpa, 'w', encoding='utf-8') as f:
        for r in rows_mpa:
            f.write(r)
    if save_mean_table and len(mean_rows) > 0:
        mean_path = os.path.join(out_dir, 'per_sample_mean_stress.csv')
        with open(mean_path, 'w', encoding='utf-8') as f:
            for r in mean_rows:
                f.write(r)

    # aggregated histogram across all samples
    if 'plt' in locals() and plt is not None and len(all_flat_errors) > 0:
        agg = np.concatenate(all_flat_errors, axis=0)
        fig = plt.figure(figsize=(5,3), dpi=150)
        plt.hist(agg, bins=bins, color='#F58518')
        plt.xlabel('Absolute error (normalized)')
        plt.ylabel('Count (all samples)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'error_hist_all.png'))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ML_DATASET')
    ap.add_argument('--ckpt', default=os.path.join('ML_CHECKPOINTS','best.pt'))
    ap.add_argument('--out', default='ML_EVAL')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--bins', type=int, default=50)
    ap.add_argument('--hist-samples', type=int, default=12, help='number of random samples to save histograms for')
    ap.add_argument('--save-all-hists', action='store_true', help='save histogram for every sample (can be many images)')
    ap.add_argument('--overlay-boundaries', action='store_true')
    ap.add_argument('--boundary-channel', type=int, default=0, help='which input channel to derive boundaries from (default: 0=E_map)')
    ap.add_argument('--sigma-scale', type=float, default=1000.0, help='scale factor to convert normalized stress to MPa')
    ap.add_argument('--export-mpa-plots', action='store_true', help='also save qualitative plots in MPa units')
    ap.add_argument('--metadata-csv', default=os.path.join('ML_DATASET','metadata','increments_map.csv'))
    ap.add_argument('--labels-dir', default='labels')
    ap.add_argument('--auto-align-boundaries', action='store_true', help='rotate/flip stress maps to match labels before drawing GB contours')
    ap.add_argument('--save-predictions', action='store_true', help='save normalized predictions for each test sample')
    ap.add_argument('--predictions-dir', default=None, help='where to store saved predictions (default: <out>/predictions)')
    ap.add_argument('--no-mean-table', action='store_true', help='disable writing per-sample mean stress table')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    test_ds = NpyPairDataset(os.path.join(args.data, 'test'))
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_ch=5, out_ch=1, base=args.base).to(device)
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['model'])

    mae, rmse = evaluate_test(model, test_ld, device)

    metrics = {'test_mae': float(mae), 'test_rmse': float(rmse)}
    with open(os.path.join(args.out, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, 'test_metrics.csv'), 'w', encoding='utf-8') as f:
        f.write('test_mae,test_rmse\n')
        f.write(f"{mae:.8f},{rmse:.8f}\n")
    print(f"Test MAE {mae:.6f} | RMSE {rmse:.6f}")

    # optional mapping for exact labels-based boundaries
    sample_to_seed = _load_sample_to_seed_map(args.metadata_csv)

    save_qualitative(
        model,
        test_ds,
        os.path.join(args.out, 'qualitative'),
        device,
        num_samples=3,
        overlay_boundaries=args.overlay_boundaries,
        boundary_channel=args.boundary_channel,
        sigma_scale=args.sigma_scale,
        export_mpa_plots=args.export_mpa_plots,
        sample_to_seed=sample_to_seed,
        labels_dir=args.labels_dir,
        auto_align_boundaries=args.auto_align_boundaries,
    )
    # per-sample statistics and (optional) histograms
    save_per_sample_stats_and_hist(
        model,
        test_ds,
        args.out,
        device,
        bins=args.bins,
        hist_samples=args.hist_samples,
        save_all_hists=args.save_all_hists,
        sigma_scale=args.sigma_scale,
        save_predictions=args.save_predictions,
        predictions_dir=args.predictions_dir,
        save_mean_table=not args.no_mean_table,
    )


if __name__ == '__main__':
    main()


