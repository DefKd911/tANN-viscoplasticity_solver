#!/usr/bin/env python3
"""
Visualize ground-truth stress (von Mises) from HDF5 alongside 4 property maps
with grain-boundary overlays.

Usage example:
  python visualize_gt_from_hdf5.py \
    --hdf5-dir simulation_results/hdf5_files \
    --metadata-csv ML_DATASET/metadata/increments_map.csv \
    --labels-dir labels \
    --dataset-root ML_DATASET \
    --seeds 1103204645,1098063911,10824797 \
    --out ML_EVAL/gt_overlays

If --seeds is omitted, use --num N to select N random seeds from hdf5-dir.
"""
import os
import re
import csv
import glob
import json
import math
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py


def compute_cauchy_stress(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute Cauchy stress sigma from F and P (1st Piola-Kirchhoff): sigma = (1/detF) * P * F^T
    Supports shapes:
      - (H,W,3,3)
      - (T,H,W,3,3) → returns (H,W,3,3) for last T
    """
    if F.ndim == 5:  # (T,H,W,3,3)
        F = F[-1]
        P = P[-1]
    if F.ndim != 4:
        raise ValueError(f"Unsupported F shape: {F.shape}")
    H, W, _, _ = F.shape
    sigma = np.zeros((H, W, 3, 3), dtype=F.dtype)
    for i in range(H):
        Fi = F[i]   # (W,3,3)
        Pi = P[i]   # (W,3,3)
        detF = np.linalg.det(Fi)  # (W,)
        detF = np.where(np.abs(detF) < 1e-12, 1e-12, detF)
        tmp = np.einsum('...ij,...kj->...ik', Pi, Fi)  # (W,3,3) = P @ F^T
        sigma[i] = tmp * (1.0 / detF)[:, None, None]
    return sigma


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    """Compute von Mises stress field from sigma (H,W,3,3)."""
    if sigma.ndim == 5:  # (T,H,W,3,3)
        sigma = sigma[-1]
    H, W, _, _ = sigma.shape
    vm = np.zeros((H, W), dtype=sigma.dtype)
    s11 = sigma[..., 0, 0]
    s22 = sigma[..., 1, 1]
    s33 = sigma[..., 2, 2]
    s12 = sigma[..., 0, 1]
    s23 = sigma[..., 1, 2]
    s31 = sigma[..., 2, 0]
    vm[:] = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2 + 6.0 * (s12 ** 2 + s23 ** 2 + s31 ** 2)))
    return vm


def _find_tensor(g: h5py.Group, keys: List[str]) -> Optional[h5py.Dataset]:
    for k in keys:
        if k in g and isinstance(g[k], h5py.Dataset):
            return g[k]
        if "output" in g and k in g["output"] and isinstance(g["output"][k], h5py.Dataset):
            return g["output"][k]
    return None


def _list_increment_groups(h5: h5py.File) -> List[str]:
    incs = []
    for k in h5.keys():
        if isinstance(h5[k], h5py.Group) and k.startswith('increment_'):
            try:
                n = int(k.split('_')[-1])
                incs.append((n, k))
            except Exception:
                pass
    incs.sort(key=lambda x: x[0])
    return [name for _, name in incs]


def load_vm_from_hdf5(h5_path: str) -> np.ndarray:
    """Robustly load a per-voxel von Mises stress map from a DAMASK HDF5.
    Strategy:
      1) Prefer stress_Cauchy if present (phase first, then homogenization)
      2) Else compute from F and P (phase or homogenization)
    Returns vm as (H,W) for the last increment.
    """
    with h5py.File(h5_path, 'r') as h5:
        # Work inside last increment if present
        root_groups: List[h5py.Group] = []
        incs = _list_increment_groups(h5)
        if len(incs) > 0:
            ginc = h5[incs[-1]]
            root_groups.append(ginc)
        else:
            root_groups.append(h5)

        candidates = []
        for root in root_groups:
            # phase level
            if 'phase' in root:
                for pid in root['phase'].keys():
                    grp = root['phase'][pid]
                    if 'mechanical' in grp:
                        candidates.append(grp['mechanical'])
                    else:
                        candidates.append(grp)
            # homogenization level
            if 'homogenization' in root:
                for hid in root['homogenization'].keys():
                    grp = root['homogenization'][hid]
                    if 'mechanical' in grp:
                        candidates.append(grp['mechanical'])
                    else:
                        candidates.append(grp)

        # Try stress_Cauchy
        for grp in candidates:
            ds = _find_tensor(grp, ['stress_Cauchy', 'sigma', 'stress'])
            if ds is None:
                continue
            arr = ds[...]
            if arr.ndim == 5:  # (T,H,W,3,3)
                sigma = arr[-1]
            elif arr.ndim == 4:  # (H,W,3,3)
                sigma = arr
            else:
                continue
            return von_mises_stress(sigma)

        # Else try F,P
        for grp in candidates:
            Fds = _find_tensor(grp, ['F', 'deformation_gradient'])
            Pds = _find_tensor(grp, ['P', 'PK1'])
            if Fds is None or Pds is None:
                continue
            F = Fds[...]
            P = Pds[...]
            # Case A: flattened grid (N,3,3)
            if F.ndim == 3 and F.shape[-2:] == (3,3) and P.ndim == 3 and P.shape[-2:] == (3,3):
                N = F.shape[0]
                s = int(round(math.sqrt(N)))
                if s*s == N:
                    F_last = F.reshape(s, s, 3, 3)
                    P_last = P.reshape(s, s, 3, 3)
                else:
                    # cannot infer 2D shape reliably
                    continue
            # Case B: (H,W,3,3) or (T,H,W,3,3)
            elif F.ndim >= 4 and F.shape[-2:] == (3,3) and P.shape[-2:] == (3,3):
                if F.ndim == 4:
                    F_last, P_last = F, P
                else:
                    F_last = F.reshape((-1,)+F.shape[-4:])[-1]
                    P_last = P.reshape((-1,)+P.shape[-4:])[-1]
            else:
                continue
            sigma = compute_cauchy_stress(F_last, P_last)
            return von_mises_stress(sigma)

    raise RuntimeError(f"Could not find stress_Cauchy or F/P to build vm in {h5_path}")


def estimate_eps11_from_hdf5(h5_path: str) -> float:
    """Return average engineering strain ε11 from the last increment.
    ε11 ≈ mean(F11) - 1.0
    """
    with h5py.File(h5_path, 'r') as h5:
        incs = _list_increment_groups(h5)
        root = h5[incs[-1]] if len(incs) > 0 else h5
        candidates = []
        if 'homogenization' in root:
            for hid in root['homogenization'].keys():
                grp = root['homogenization'][hid]
                if 'mechanical' in grp:
                    candidates.append(grp['mechanical'])
                else:
                    candidates.append(grp)
        if 'phase' in root:
            for pid in root['phase'].keys():
                grp = root['phase'][pid]
                if 'mechanical' in grp:
                    candidates.append(grp['mechanical'])
                else:
                    candidates.append(grp)
        for g in candidates:
            if 'F' in g:
                F = g['F'][...]
                if F.ndim == 3:
                    F11 = F[:, 0, 0]
                elif F.ndim == 4:
                    F11 = F[-1, ..., 0, 0].reshape(-1)
                else:
                    continue
                return float(F11.mean() - 1.0)
    # Fallback to nominal 0.004 if nothing found
    return 0.004


def load_labels(labels_dir: str, seed: str) -> Optional[np.ndarray]:
    p = os.path.join(labels_dir, f'seed{seed}.npy')
    if os.path.isfile(p):
        try:
            arr = np.load(p)
            if arr.ndim == 2:
                return arr
        except Exception:
            return None
    return None


def compute_gb_mask_from_labels(lbl: np.ndarray) -> np.ndarray:
    H, W = lbl.shape
    m = np.zeros((H, W), dtype=bool)
    m[:, :-1] |= (lbl[:, 1:] != lbl[:, :-1])
    m[:-1, :] |= (lbl[1:, :] != lbl[:-1, :])
    return m


def _edges_from_scalar_field(arr: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    H, W = arr.shape
    m = np.zeros((H, W), dtype=bool)
    m[:, :-1] |= np.abs(arr[:, 1:] - arr[:, :-1]) > tol
    m[:-1, :] |= np.abs(arr[1:, :] - arr[:-1, :]) > tol
    return m


def _apply_transform(img: np.ndarray, t: str) -> np.ndarray:
    if t == 'identity':
        return img
    if t == 'rot90':
        return np.rot90(img, 1)
    if t == 'rot180':
        return np.rot90(img, 2)
    if t == 'rot270':
        return np.rot90(img, 3)
    if t == 'flipud':
        return np.flipud(img)
    if t == 'fliplr':
        return np.fliplr(img)
    if t == 'rot90_fliplr':
        return np.fliplr(np.rot90(img, 1))
    if t == 'rot90_flipud':
        return np.flipud(np.rot90(img, 1))
    return img


def auto_align_to_labels(vm_map: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, str]:
    """Try D4 transforms to align vm_map to labels using boundary-edge overlap.
    Returns (aligned_vm, transform_name)."""
    lb = compute_gb_mask_from_labels(labels)
    # build edges from vm via normalized diffs thresholded at 5% of range
    vmin, vmax = float(vm_map.min()), float(vm_map.max())
    thr = max(1e-12, 0.05 * (vmax - vmin))
    vm_edges = _edges_from_scalar_field(vm_map, tol=thr)
    transforms = ['identity', 'rot90', 'rot180', 'rot270', 'flipud', 'fliplr', 'rot90_fliplr', 'rot90_flipud']
    best_score = -1.0
    best_t = 'identity'
    for t in transforms:
        ve = _apply_transform(vm_edges, t)
        if ve.shape != lb.shape:
            continue
        inter = np.logical_and(ve, lb).sum()
        union = np.logical_or(ve, lb).sum()
        score = inter / max(1, union)
        if score > best_score:
            best_score = score
            best_t = t
    return _apply_transform(vm_map, best_t), best_t


def find_labels_path(labels_dir: str, seed: str) -> Optional[str]:
    candidates = [
        os.path.join(labels_dir, f"labels_seed{seed}.npy"),
        os.path.join(labels_dir, f"seed{seed}.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_props_path(props_dir: str, seed: str) -> Optional[str]:
    candidates = [
        os.path.join(props_dir, f"props_seed{seed}.npy"),
        os.path.join(props_dir, f"seed{seed}.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def per_pixel_maps(labels: np.ndarray, props: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if labels.ndim != 2:
        raise ValueError("labels must be 2D (H,W)")
    if props.ndim == 0 and isinstance(props.item(), dict):
        d = props.item()
        props = np.stack([np.asarray(d['E']), np.asarray(d['nu']), np.asarray(d['xi0']), np.asarray(d['h0'])], axis=1)
    if props.ndim != 2 or props.shape[1] < 4:
        raise ValueError("props must be (G,4) with [E, nu, xi0, h0]")
    max_label = labels.max()
    if max_label >= props.shape[0]:
        raise ValueError(f"labels contain grain id {max_label} but props only has {props.shape[0]} rows")
    E_map = props[labels, 0]
    nu_map = props[labels, 1]
    xi0_map = props[labels, 2]
    h0_map = props[labels, 3]
    return E_map, nu_map, xi0_map, h0_map


def load_sample_to_seed_map(metadata_csv: str) -> Dict[str, Tuple[str, str]]:
    """Return mapping: seed -> (split, sample_name). If multiple rows per seed, pick last (max increment)."""
    mapping: Dict[str, Tuple[str, str]] = {}
    if not os.path.isfile(metadata_csv):
        return mapping
    try:
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return mapping

    # Group by seed and choose max increment (if available)
    by_seed: Dict[str, List[dict]] = {}
    for r in rows:
        seed = None
        for k in r.keys():
            if 'seed' in k.lower():
                m = re.search(r'(\d+)', str(r[k]))
                if m:
                    seed = m.group(1)
                    break
        if seed is None:
            continue
        by_seed.setdefault(seed, []).append(r)

    for seed, rs in by_seed.items():
        # get increment if present, else fallback to last row
        best = rs[-1]
        best_inc = -1
        for r in rs:
            inc = -1
            for k, v in r.items():
                if 'increment' in k.lower():
                    try:
                        inc = int(v)
                    except Exception:
                        inc = -1
                    break
            if inc >= best_inc:
                best_inc = inc
                best = r
        # detect split and sample
        split = best.get('split', '').strip() if best.get('split') else ''
        sample = best.get('sample', '').strip() if best.get('sample') else ''
        if not sample:
            # try to infer sample from any column value like 'sample_01234.npy'
            for v in best.values():
                s = str(v)
                if 'sample_' in s:
                    sample = os.path.basename(s).replace('.npy','')
                    break
        if split and sample:
            mapping[seed] = (split, sample)
    return mapping


def load_props_from_dataset(dataset_root: str, split: str, sample: str, prop_channels: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = os.path.join(dataset_root, split, 'inputs', f'{sample}.npy')
    X = np.load(p).astype(np.float32)  # (H,W,C)
    c0, c1, c2, c3 = prop_channels
    E = X[..., c0]
    nu = X[..., c1]
    xi0 = X[..., c2]
    h0 = X[..., c3]
    return E, nu, xi0, h0


def draw_panels(
    seed: str,
    vm: np.ndarray,
    props: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    labels: Optional[np.ndarray],
    out_dir: str,
    sigma_scale: float = 1e-6,
    overlay_boundaries: bool = False,
    elastic_vm_mpa: Optional[np.ndarray] = None,
    delta_vm_mpa: Optional[np.ndarray] = None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return

    E, nu, xi0, h0 = props
    # Convert to display units
    E_disp = E / 1e9      # GPa
    nu_disp = nu          # -
    xi0_disp = xi0 / 1e6  # MPa
    h0_disp = h0 / 1e9    # GPa
    # Resize checks
    H, W = vm.shape
    for name, arr in [('E', E), ('nu', nu), ('xi0', xi0), ('h0', h0)]:
        if arr.shape != (H, W):
            raise ValueError(f"Shape mismatch for {name}: {arr.shape} vs vm {vm.shape}")
    bmask = compute_gb_mask_from_labels(labels) if (overlay_boundaries and labels is not None) else None

    vmin = float(vm.min())
    vmax = float(vm.max())
    vmin_mpa = vmin * sigma_scale
    vmax_mpa = vmax * sigma_scale

    # Determine column count (5 base + optional 2)
    num_cols = 5 + (2 if elastic_vm_mpa is not None and delta_vm_mpa is not None else 0)
    fig = plt.figure(figsize=(3.0*num_cols, 3.2), dpi=150)
    grids = [
        (E_disp, 'E (GPa)', 'viridis', None, None),
        (nu_disp, 'nu (-)', 'viridis', None, None),
        (xi0_disp, 'xi0 (MPa)', 'viridis', None, None),
        (h0_disp, 'h0 (GPa)', 'viridis', None, None),
        (vm, 'σvM (MPa)', 'viridis', vmin_mpa, vmax_mpa),
    ]
    if elastic_vm_mpa is not None and delta_vm_mpa is not None:
        # Elastic estimate shares σvM color limits for direct comparison
        grids.append((elastic_vm_mpa, 'σvM_el (MPa)', 'viridis', vmin_mpa, vmax_mpa))
        # Deviation uses symmetric limits
        d_abs = float(np.max(np.abs(delta_vm_mpa)))
        grids.append((delta_vm_mpa, 'ΔσvM (MPa)', 'RdBu_r', -d_abs, d_abs))

    for i, (arr, title, cmap, vmin_, vmax_) in enumerate(grids, start=1):
        ax = plt.subplot(1, num_cols, i)
        data = arr * sigma_scale if title == 'σvM (MPa)' else arr
        if title == 'σvM (MPa)':
            im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_, origin='lower', interpolation='nearest')
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_, origin='lower', interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if overlay_boundaries and bmask is not None:
            ax.contour(bmask.astype(float), levels=[0.5], colors='white', linewidths=0.5, alpha=0.7, origin='lower')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    # Filename adapts if elastic panels are included
    fname = f'seed{seed}_gt_5panel.png' if num_cols == 5 else f'seed{seed}_gt_7panel.png'
    plt.savefig(os.path.join(out_dir, fname))
    plt.close(fig)


def find_hdf5_for_seed(hdf5_dir: str, seed: str) -> str:
    p = os.path.join(hdf5_dir, f'seed{seed}.hdf5')
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"HDF5 for seed {seed} not found at {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5-dir', default=os.path.join('simulation_results','hdf5_files'))
    ap.add_argument('--metadata-csv', default=os.path.join('ML_DATASET','metadata','increments_map.csv'))
    ap.add_argument('--labels-dir', default='labels')
    ap.add_argument('--dataset-root', default='ML_DATASET')
    ap.add_argument('--props-dir', default='props')
    ap.add_argument('--seeds', default='')
    ap.add_argument('--num', type=int, default=5)
    ap.add_argument('--prop-channels', default='0,1,2,3', help='indices of E,nu,xi0,h0 in inputs')
    # Scale σvM (stored in Pa) to display units. Default converts Pa → MPa.
    ap.add_argument('--sigma-scale', type=float, default=1e-6)
    ap.add_argument('--out', default=os.path.join('ML_EVAL','gt_overlays'))
    ap.add_argument('--overlay-boundaries', action='store_true')
    ap.add_argument('--auto-align', action='store_true', help='auto-rotate/flip σvM to align with labels')
    ap.add_argument('--elastic-check', action='store_true', help='add Hooke-law estimate and deviation panels')
    args = ap.parse_args()

    # collect seeds
    seeds: List[str] = []
    if args.seeds.strip():
        seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    else:
        # sample randomly from available files
        files = sorted(glob.glob(os.path.join(args.hdf5_dir, 'seed*.hdf5')))
        file_seeds = [re.search(r'seed(\d+)\.hdf5$', os.path.basename(f)).group(1) for f in files]
        seeds = random.sample(file_seeds, k=min(args.num, len(file_seeds)))

    # mapping seed -> (split, sample)
    seed_to_split_sample = load_sample_to_seed_map(args.metadata_csv)

    # parse channels
    ch = tuple(int(x) for x in args.prop_channels.split(','))
    if len(ch) != 4:
        raise ValueError('--prop-channels must provide 4 comma-separated integers')

    for seed in seeds:
        try:
            h5_path = find_hdf5_for_seed(args.hdf5_dir, seed)
            vm = load_vm_from_hdf5(h5_path)

            # First try labels+props direct route (no metadata needed)
            E = nu = xi0 = h0 = None
            lbl = None
            lbl_path = find_labels_path(args.labels_dir, seed)
            props_path = find_props_path(args.props_dir, seed)
            if lbl_path and props_path:
                labels = np.load(lbl_path, allow_pickle=True)
                props = np.load(props_path, allow_pickle=True)
                E, nu, xi0, h0 = per_pixel_maps(labels, props)
            else:
                # fallback to dataset mapping (values are normalized → de-normalize)
                if seed not in seed_to_split_sample:
                    raise RuntimeError(f"Seed {seed} not found in metadata {args.metadata_csv}")
                split, sample = seed_to_split_sample[seed]
                E_n, nu_n, xi0_n, h0_n = load_props_from_dataset(args.dataset_root, split, sample, ch)
                # invert normalization used in dataset build
                E = E_n * 250e9 + 50e9
                nu = nu_n * 0.2 + 0.2
                xi0 = xi0_n * 250e6 + 50e6
                h0 = h0_n * 50e9

            # labels for GB overlay / alignment (optional)
            if (args.overlay_boundaries or args.auto_align) and lbl is None and lbl_path:
                lbl = np.load(lbl_path)

            # auto alignment of vm to labels if requested
            transform_used = 'identity'
            if args.auto_align and lbl is not None:
                vm_aligned, transform_used = auto_align_to_labels(vm, lbl)
                vm = vm_aligned

            # sanity shape check
            if (E.shape != vm.shape) or (nu.shape != vm.shape) or (xi0.shape != vm.shape) or (h0.shape != vm.shape):
                raise RuntimeError(f"Shape mismatch for seed {seed}: props {E.shape,nu.shape,xi0.shape,h0.shape} vs vm {vm.shape}")

            # Print σvM stats in MPa
            vm_min_mpa = float(vm.min()) * 1e-6
            vm_max_mpa = float(vm.max()) * 1e-6
            vm_mean_mpa = float(vm.mean()) * 1e-6

            elastic_vm_mpa = None
            delta_vm_mpa = None
            if args.elastic_check:
                # Average ε11 from HDF5
                eps11 = estimate_eps11_from_hdf5(h5_path)
                # Elastic estimate in Pa, then convert to MPa
                # For uniaxial tension: σ_vM = σ11 = E × ε11 (NOT E × ε11 / (1+ν))
                elastic_vm_mpa = (E * eps11) / 1e6
                delta_vm_mpa = (vm * 1e-6) - elastic_vm_mpa

            draw_panels(
                seed,
                vm,
                (E, nu, xi0, h0),
                lbl,
                args.out,
                sigma_scale=args.sigma_scale,
                overlay_boundaries=args.overlay_boundaries,
                elastic_vm_mpa=elastic_vm_mpa,
                delta_vm_mpa=delta_vm_mpa,
            )
            outf = f'seed{seed}_gt_7panel.png' if elastic_vm_mpa is not None else f'seed{seed}_gt_5panel.png'
            print(f"Saved: {os.path.join(args.out, outf)} (transform={transform_used}) | sigma_vM MPa: min={vm_min_mpa:.2f}, mean={vm_mean_mpa:.2f}, max={vm_max_mpa:.2f}")
        except Exception as e:
            print(f"Failed for seed {seed}: {e}")


if __name__ == '__main__':
    main()


