#!/usr/bin/env python3
"""
Analyze per-grain stress-strain curves for test samples:
- Extract GT stress-strain curves from HDF5 files
- Extract predicted stress-strain curves from ML predictions
- Compare predicted vs actual for each grain in each test sample

Usage:
  python analyze_test_per_grain_ss_curves.py \
    --data ML_DATASET \
    --predictions ML_EVAL/predictions \
    --hdf5-dir simulation_results/hdf5_files \
    --labels-dir labels \
    --props-dir props \
    --out ML_EVAL/per_grain_test_analysis \
    --seeds 1099931136,1098426137,1098322009,1098694165,1102020331
"""
import os
import csv
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
try:
    from scipy.interpolate import interp1d
    from scipy.ndimage import binary_dilation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, interpolation will use numpy")
    # Fallback for binary_dilation
    def binary_dilation(arr, iterations=1):
        # Simple fallback - just return the array
        return arr

# Tensor loading utilities (reused from build_ml_dataset.py)
SIGMA_DATASETS = [
    "mechanical/output/stress_Cauchy",
    "mechanical/stress_Cauchy",
    "mechanical/output/sigma",
    "mechanical/sigma",
]

F_DATASETS = [
    "mechanical/output/F",
    "mechanical/F",
]

P_DATASETS = [
    "mechanical/output/P",
    "mechanical/P",
]


def _reshape_tensor(arr: np.ndarray) -> np.ndarray:
    """Return array shaped (N,3,3) regardless of original grid layout."""
    if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        return arr
    if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    if arr.ndim == 5 and arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    raise ValueError(f"Unexpected tensor shape {arr.shape}")


def _load_dataset(f: h5py.File, base: str, relative_paths: List[str]) -> Optional[np.ndarray]:
    for rel in relative_paths:
        path = f"{base}/{rel}"
        if path in f:
            return f[path][()]
    return None


def load_sigma_tensor(f: h5py.File, base: str) -> Optional[np.ndarray]:
    arr = _load_dataset(f, base, SIGMA_DATASETS)
    if arr is None:
        return None
    return _reshape_tensor(arr)


def load_sigma_from_F_P(f: h5py.File, base: str) -> np.ndarray:
    Farr = _load_dataset(f, base, F_DATASETS)
    Parr = _load_dataset(f, base, P_DATASETS)
    if Farr is None or Parr is None:
        raise KeyError(f"No F/P tensors found under {base}")
    F = _reshape_tensor(Farr)
    P = _reshape_tensor(Parr)
    return compute_cauchy_from_F_P(F, P)


def compute_cauchy_from_F_P(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute Cauchy stress from F and P for homogenization fields."""
    if F.ndim != 3 or P.ndim != 3 or F.shape[1:] != (3,3) or P.shape[1:] != (3,3):
        raise ValueError(f"Unexpected shapes F={F.shape}, P={P.shape}")
    N = F.shape[0]
    sigma = np.zeros_like(P)
    for i in range(N):
        Fi = F[i]
        Pi = P[i]
        detF = np.linalg.det(Fi)
        if abs(detF) < 1e-14:
            raise ValueError("det(F) ~ 0 encountered")
        sigma[i] = (1.0/detF) * (Pi @ Fi.T)
    return sigma


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    """Compute von Mises stress field from sigma (N,3,3) -> (N,)."""
    N = sigma.shape[0]
    vm = np.zeros(N, dtype=np.float64)
    I = np.eye(3)
    for i in range(N):
        s = sigma[i]
        s_dev = s - np.trace(s)/3.0 * I
        vm[i] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
    return vm


def load_vm_field(f: h5py.File, base: str) -> np.ndarray:
    """Return von Mises field (N,) for a given increment base path."""
    sigma = load_sigma_tensor(f, base)
    if sigma is None:
        sigma = load_sigma_from_F_P(f, base)
    return von_mises_stress(sigma)


def extract_strain_from_F(F: np.ndarray) -> np.ndarray:
    """Extract engineering strain ε11 from F: ε11 = F11 - 1"""
    if F.ndim != 3 or F.shape[-2:] != (3, 3):
        raise ValueError(f"F must be (N, 3, 3), got {F.shape}")
    return F[:, 0, 0] - 1.0


def _list_increment_groups(h5: h5py.File) -> List[str]:
    """List all increment groups in sorted order."""
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


def load_gt_per_grain_curves(h5_path: str, labels: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load ground truth stress-strain curves per grain from HDF5.
    Returns: {grain_id: {'strain': array, 'stress_vm': array}}
    """
    H, W = labels.shape
    N_voxels = H * W
    unique_grains = np.unique(labels)
    
    # Initialize storage
    grain_data = {int(gid): {'strain': [], 'stress_vm': []} for gid in unique_grains}
    
    with h5py.File(h5_path, 'r') as f:
        increments = _list_increment_groups(f)
        
        for inc_name in increments:
            base = f"{inc_name}/homogenization/h0"
            
            try:
                # Load stress
                vm = load_vm_field(f, base)
                
                # Load F for strain
                Farr = _load_dataset(f, base, F_DATASETS)
                if Farr is None:
                    continue
                F = _reshape_tensor(Farr)
                strain_11 = extract_strain_from_F(F)
                
                # Reshape to spatial grid
                if vm.shape[0] != N_voxels:
                    if vm.shape[0] == H * W:
                        vm_2d = vm.reshape(H, W)
                        strain_2d = strain_11.reshape(H, W)
                    else:
                        continue
                else:
                    vm_2d = vm.reshape(H, W)
                    strain_2d = strain_11.reshape(H, W)
                
                # Average per grain
                for gid in unique_grains:
                    mask = (labels == gid)
                    if mask.sum() > 0:
                        grain_data[int(gid)]['strain'].append(np.mean(strain_2d[mask]))
                        grain_data[int(gid)]['stress_vm'].append(np.mean(vm_2d[mask]) / 1e6)  # Pa to MPa
                        
            except Exception as e:
                print(f"Warning: Failed to load increment {inc_name}: {e}")
                continue
    
    # Convert lists to arrays
    for gid in grain_data:
        if len(grain_data[gid]['strain']) > 0:
            grain_data[gid]['strain'] = np.array(grain_data[gid]['strain'])
            grain_data[gid]['stress_vm'] = np.array(grain_data[gid]['stress_vm'])
        else:
            grain_data[gid]['strain'] = np.array([])
            grain_data[gid]['stress_vm'] = np.array([])
    
    return grain_data


def load_predicted_per_grain_curves(
    sample_name: str,
    predictions_dir: str,
    data_root: str,
    labels: np.ndarray,
    metadata: Dict[str, Dict],
    h5_path: Optional[str] = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load predicted stress-strain curves per grain from ML predictions.
    Need to load all increments for the same seed and reconstruct the curve.
    Uses HDF5 to get actual strain values if available, otherwise estimates from increment index.
    """
    H, W = labels.shape
    unique_grains = np.unique(labels)
    
    # Get seed and find all samples for this seed
    if sample_name not in metadata:
        return {}
    
    seed = metadata[sample_name]['seed']
    split = metadata[sample_name]['split']
    
    # Find all samples for this seed in test split
    seed_samples = []
    for sname, info in metadata.items():
        if info['seed'] == seed and info['split'] == split:
            seed_samples.append((sname, info['increment_t']))
    
    # Sort by increment
    seed_samples.sort(key=lambda x: x[1])
    
    if len(seed_samples) == 0:
        return {}
    
    # Try to get actual strain values from HDF5
    strain_map = {}  # increment_t -> strain value
    if h5_path and os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                increments = _list_increment_groups(f)
                for inc_name in increments:
                    try:
                        inc_num = int(inc_name.split('_')[-1])
                        base = f"{inc_name}/homogenization/h0"
                        Farr = _load_dataset(f, base, F_DATASETS)
                        if Farr is not None:
                            F = _reshape_tensor(Farr)
                            strain_11 = extract_strain_from_F(F)
                            # Use mean strain across all voxels
                            strain_map[inc_num] = np.mean(strain_11)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: Could not load strain from HDF5: {e}")
    
    # If no HDF5 strain, estimate from increment index
    if len(strain_map) == 0:
        max_inc = max([info['increment_t'] for info in metadata.values() if info['seed'] == seed])
        # Assuming strain goes from 0 to 0.004 (0.4%)
        for inc_t in range(max_inc + 2):
            strain_map[inc_t] = 0.004 * inc_t / (max_inc + 1)
    
    # Initialize storage
    grain_data = {int(gid): {'strain': [], 'stress_vm': []} for gid in unique_grains}
    
    # Load initial stress from first input
    first_sample, first_inc = seed_samples[0]
    input_path = os.path.join(data_root, split, 'inputs', f"{first_sample}.npy")
    if os.path.exists(input_path):
        X = np.load(input_path)  # (H, W, 5)
        # Channel 4 is sigma_vM(t)
        sigma_t = X[:, :, 4] * 1000.0  # Denormalize
        
        # Strain at increment 0
        strain_0 = strain_map.get(first_inc, 0.0)
        
        for gid in unique_grains:
            mask = (labels == gid)
            if mask.sum() > 0:
                grain_data[int(gid)]['strain'].append(strain_0)
                grain_data[int(gid)]['stress_vm'].append(np.mean(sigma_t[mask]))
    
    # Load predictions for each increment
    for sname, inc_t in seed_samples:
        # Load prediction (this is stress at increment t+1)
        pred_path = os.path.join(predictions_dir, f"{sname}.npy")
        if not os.path.exists(pred_path):
            continue
        
        pred = np.load(pred_path)
        if pred.ndim == 3:
            pred = pred[0]  # Remove channel dimension if present
        
        # Denormalize: sigma_vM was normalized as (sigma_vM[Pa]/1e6)/1000.0
        # So to denormalize: pred_mpa = pred * 1000.0
        pred_mpa = pred * 1000.0
        
        # Strain at increment t+1
        strain_tp1 = strain_map.get(inc_t + 1, strain_map.get(inc_t, 0.0) + 0.004 / len(seed_samples))
        
        # Average per grain
        for gid in unique_grains:
            mask = (labels == gid)
            if mask.sum() > 0:
                grain_data[int(gid)]['strain'].append(strain_tp1)
                grain_data[int(gid)]['stress_vm'].append(np.mean(pred_mpa[mask]))
    
    # Convert to arrays and sort by strain
    for gid in grain_data:
        if len(grain_data[gid]['strain']) > 0:
            strains = np.array(grain_data[gid]['strain'])
            stresses = np.array(grain_data[gid]['stress_vm'])
            # Sort by strain
            sort_idx = np.argsort(strains)
            grain_data[gid]['strain'] = strains[sort_idx]
            grain_data[gid]['stress_vm'] = stresses[sort_idx]
        else:
            grain_data[gid]['strain'] = np.array([])
            grain_data[gid]['stress_vm'] = np.array([])
    
    return grain_data


def load_metadata(metadata_csv: str) -> Dict[str, Dict]:
    """Load sample metadata from CSV."""
    metadata = {}
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_idx = int(row['sample_idx'])
            sample_name = f"sample_{sample_idx:05d}"
            metadata[sample_name] = {
                'seed': row['seed'],
                'split': row['split'],
                'increment_t': int(row['increment_t'])
            }
    return metadata


def find_labels_path(seed: str, labels_dir: str) -> Optional[str]:
    """Find labels file for a seed."""
    # Handle seeds with or without "seed" prefix
    seed_clean = seed.replace('seed', '') if seed.startswith('seed') else seed
    
    candidates = [
        os.path.join(labels_dir, f"labels_seed{seed_clean}.npy"),
        os.path.join(labels_dir, f"seed{seed_clean}.npy"),
        os.path.join(labels_dir, f"labels_{seed_clean}.npy"),
        os.path.join(labels_dir, f"{seed}.npy"),  # In case seed already has prefix
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_props_path(seed: str, props_dir: str) -> Optional[str]:
    """Find props file for a seed."""
    # Handle seeds with or without "seed" prefix
    seed_clean = seed.replace('seed', '') if seed.startswith('seed') else seed
    
    candidates = [
        os.path.join(props_dir, f"props_seed{seed_clean}.npy"),
        os.path.join(props_dir, f"seed{seed_clean}.npy"),
        os.path.join(props_dir, f"props_{seed_clean}.npy"),
        os.path.join(props_dir, f"{seed}.npy"),  # In case seed already has prefix
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def classify_grain_behavior(gt_data: Dict[str, np.ndarray], props: np.ndarray, grain_id: int, 
                           yield_threshold: float = 0.1) -> Tuple[str, float]:
    """
    Classify grain as 'elastic' or 'plastic' based on stress-strain curve.
    
    CLASSIFICATION CRITERIA:
    ------------------------
    A grain is classified as PLASTIC if ANY of the following conditions are met:
    
    1. **Yield Stress Criterion**: 
       - Maximum stress exceeds yield stress (ξ₀) by >10%
       - Formula: max(σ_vM) > 1.1 × ξ₀
       - Physical meaning: Grain has yielded and entered plastic regime
       
    2. **Deviation from Elastic Criterion**:
       - Maximum deviation from elastic line exceeds threshold (default: 10%)
       - Formula: max(|σ_vM - σ_elastic| / σ_elastic) > yield_threshold
       - Where: σ_elastic = E × ε (Hooke's law)
       - Physical meaning: Stress significantly deviates from linear elastic behavior
       
    A grain is classified as ELASTIC if:
    - Maximum stress ≤ 1.1 × ξ₀ (hasn't yielded)
    - AND maximum deviation ≤ yield_threshold (follows elastic line)
    
    PARAMETERS:
    -----------
    yield_threshold: float (default=0.1)
        Maximum allowed deviation from elastic line (10% = 0.1)
        If deviation exceeds this, grain is classified as plastic
        
    RETURNS:
    --------
    (classification, deviation_percentage)
    - classification: 'elastic', 'plastic', or 'unknown'
    - deviation_percentage: Maximum deviation from elastic line (%)
    
    PHYSICAL INTERPRETATION:
    ------------------------
    - **Elastic grains**: Follow Hooke's law (σ = E×ε), haven't yielded
    - **Plastic grains**: Have yielded, show hardening behavior (σ > E×ε)
    - **Unknown**: Insufficient data to classify
    
    EXAMPLE:
    --------
    Grain with ξ₀ = 200 MPa, E = 200 GPa:
    - If max stress = 150 MPa → ELASTIC (hasn't reached yield)
    - If max stress = 250 MPa → PLASTIC (exceeded yield stress)
    - If stress follows E×ε line → ELASTIC
    - If stress deviates >10% from E×ε → PLASTIC
    """
    if len(gt_data['strain']) == 0 or grain_id >= props.shape[0]:
        return 'unknown', 0.0
    
    E = props[grain_id, 0] / 1e9  # GPa
    xi0 = props[grain_id, 2] / 1e6  # MPa
    
    # Compute elastic estimate using Hooke's law: σ = E × ε
    strain = gt_data['strain']
    stress_gt = gt_data['stress_vm']
    stress_el = (E * 1e9 * strain) / 1e6  # Convert to MPa: (E in Pa) × ε / 1e6
    
    # Calculate deviation from elastic line (percentage)
    # Avoid division by zero for small elastic stresses
    deviation = np.abs(stress_gt - stress_el) / (stress_el + 1e-6)
    max_deviation = np.max(deviation)
    
    # Criterion 1: Check if stress exceeds yield stress (ξ₀)
    max_stress = np.max(stress_gt) if len(stress_gt) > 0 else 0
    has_yielded = max_stress > xi0 * 1.1  # 10% margin to account for numerical noise
    
    # Criterion 2: Check if deviation from elastic line exceeds threshold
    significant_deviation = max_deviation > yield_threshold
    
    # Classification logic
    if has_yielded or significant_deviation:
        return 'plastic', float(max_deviation * 100)  # Return as percentage
    else:
        return 'elastic', float(max_deviation * 100)


def plot_per_grain_comparison(
    grain_id: int,
    gt_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    props: np.ndarray,
    out_path: str,
    behavior: str = 'unknown'
):
    """Plot predicted vs actual stress-strain curve for a single grain."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ground truth
    if len(gt_data['strain']) > 0:
        ax.plot(gt_data['strain'] * 100, gt_data['stress_vm'], 
               'o-', color='#2E86AB', linewidth=2, markersize=6, 
               label='Ground Truth (DAMASK)', alpha=0.8)
    
    # Prediction
    if len(pred_data['strain']) > 0:
        ax.plot(pred_data['strain'] * 100, pred_data['stress_vm'], 
               's--', color='#A23B72', linewidth=2, markersize=6, 
               label='Predicted (ML)', alpha=0.8)
    
    # Elastic estimate
    if len(gt_data['strain']) > 0 and grain_id < props.shape[0]:
        E = props[grain_id, 0] / 1e9  # GPa
        nu = props[grain_id, 1]
        xi0 = props[grain_id, 2] / 1e6  # MPa
        h0 = props[grain_id, 3] / 1e9  # GPa
        
        # Elastic line: sigma = E * epsilon
        strain_max = gt_data['strain'].max() if len(gt_data['strain']) > 0 else 0.004
        strain_el = np.linspace(0, strain_max, 100)
        stress_el = (E * 1e9 * strain_el) / 1e6  # Convert to MPa
        
        ax.plot(strain_el * 100, stress_el, '--', color='gray', 
               linewidth=1.5, alpha=0.6, label=f'Elastic (E={E:.0f} GPa)')
        
        # Yield stress line
        if xi0 > 0:
            ax.axhline(y=xi0, color='orange', linestyle=':', 
                      linewidth=1.5, alpha=0.6, label=f'Yield (ξ₀={xi0:.0f} MPa)')
    
    # Add behavior classification to title
    behavior_label = f" [{behavior.upper()}]" if behavior != 'unknown' else ""
    
    ax.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Von Mises Stress (MPa)', fontsize=12, fontweight='bold')
    ax.set_title(f'Grain {grain_id} Stress-Strain Curve{behavior_label}\n' +
                f'E={E:.0f} GPa, ν={nu:.2f}, ξ₀={xi0:.0f} MPa, h₀={h0:.1f} GPa',
                fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_microstructure_with_properties(
    labels: np.ndarray,
    props: np.ndarray,
    grain_behaviors: Dict[int, str],
    seed: str,
    out_path: str,
    h5_path: Optional[str] = None,
    pred_stress_map: Optional[np.ndarray] = None
):
    """Plot microstructure with grain properties and behavior classification.
    
    Args:
        labels: Grain labels (H, W)
        props: Grain properties (G, 4) [E, nu, xi0, h0]
        grain_behaviors: Dict mapping grain_id -> 'elastic'/'plastic'/'unknown'
        seed: Seed identifier
        out_path: Output file path
        h5_path: Path to HDF5 file for loading GT stress
        pred_stress_map: Predicted stress map (H, W) in MPa (optional)
    """
    H, W = labels.shape
    unique_grains = np.unique(labels)
    
    # Helper function for auto-alignment (from visualize_gt_from_hdf5.py)
    def _auto_align_field_to_labels(field: np.ndarray, labels: np.ndarray):
        """Align scalar field to labels by maximizing edge overlap."""
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
        
        lbl_edges = _compute_boundary_mask_from_labels(labels)
        field_range = float(field.max() - field.min())
        tol = max(1e-6, 0.05 * field_range)
        transforms = ['identity', 'rot90', 'rot180', 'rot270', 'flipud', 'fliplr', 'rot90_fliplr', 'rot90_flipud']
        best_score = -1.0
        best_t = 'identity'
        best_field = field
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
    
    # Load von Mises stress from HDF5 if available (GT stress)
    vm_stress_map_gt = None
    if h5_path and os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                increments = _list_increment_groups(f)
                if len(increments) > 0:
                    # Use last increment
                    last_inc = increments[-1]
                    base = f"{last_inc}/homogenization/h0"
                    vm_flat = load_vm_field(f, base)
                    if vm_flat.shape[0] == H * W:
                        vm_stress_map_gt_raw = vm_flat.reshape(H, W) / 1e6  # Convert to MPa
                        # Auto-align to labels to ensure same microstructure
                        vm_stress_map_gt, transform_gt = _auto_align_field_to_labels(vm_stress_map_gt_raw, labels)
                        if transform_gt != 'identity':
                            print(f"  Aligned GT stress using transform: {transform_gt}")
        except Exception as e:
            print(f"Warning: Could not load von Mises stress from HDF5: {e}")
            vm_stress_map_gt = None
    
    # Determine common color scale for stress maps
    stress_vmin = None
    stress_vmax = None
    if vm_stress_map_gt is not None:
        stress_vmin = float(vm_stress_map_gt.min())
        stress_vmax = float(vm_stress_map_gt.max())
    if pred_stress_map is not None:
        pred_min = float(pred_stress_map.min())
        pred_max = float(pred_stress_map.max())
        if stress_vmin is None:
            stress_vmin = pred_min
            stress_vmax = pred_max
        else:
            stress_vmin = min(stress_vmin, pred_min)
            stress_vmax = max(stress_vmax, pred_max)
    
    # Create property maps
    E_map = np.zeros((H, W))
    nu_map = np.zeros((H, W))
    xi0_map = np.zeros((H, W))
    h0_map = np.zeros((H, W))
    behavior_map = np.zeros((H, W))  # 0=elastic, 1=plastic, 2=unknown
    
    # Calculate grain centroids for labeling
    grain_centroids = {}
    
    for gid in unique_grains:
        gid = int(gid)
        mask = (labels == gid)
        if gid < props.shape[0]:
            E_map[mask] = props[gid, 0] / 1e9  # GPa
            nu_map[mask] = props[gid, 1]
            xi0_map[mask] = props[gid, 2] / 1e6  # MPa
            h0_map[mask] = props[gid, 3] / 1e9  # GPa
            
            behavior = grain_behaviors.get(gid, 'unknown')
            if behavior == 'plastic':
                behavior_map[mask] = 1
            elif behavior == 'elastic':
                behavior_map[mask] = 0
            else:
                behavior_map[mask] = 2
        
        # Calculate centroid for this grain
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            grain_centroids[gid] = (centroid_x, centroid_y)
    
    # Determine common color scales for property maps (use actual data ranges)
    E_vmin, E_vmax = float(E_map.min()), float(E_map.max())
    nu_vmin, nu_vmax = float(nu_map.min()), float(nu_map.max())
    xi0_vmin, xi0_vmax = float(xi0_map.min()), float(xi0_map.max())
    h0_vmin, h0_vmax = float(h0_map.min()), float(h0_map.max())
    
    # Ensure non-zero ranges
    if E_vmax == E_vmin:
        E_vmin, E_vmax = 50.0, 300.0
    if nu_vmax == nu_vmin:
        nu_vmin, nu_vmax = 0.2, 0.4
    if xi0_vmax == xi0_vmin:
        xi0_vmin, xi0_vmax = 50.0, 300.0
    if h0_vmax == h0_vmin:
        h0_vmin, h0_vmax = 0.0, 50.0
    
    # Create figure with subplots (3x4 grid to accommodate both GT and predicted stress)
    # Or 4x3 if we want to keep it compact
    fig = plt.figure(figsize=(20, 15))
    
    # Grain boundaries (compute once, use for all maps)
    boundaries = np.zeros_like(labels, dtype=bool)
    boundaries[:, :-1] |= (labels[:, 1:] != labels[:, :-1])
    boundaries[:-1, :] |= (labels[1:, :] != labels[:-1, :])
    if HAS_SCIPY:
        boundaries = binary_dilation(boundaries, iterations=1)
    
    # Helper function to add grain labels consistently
    def add_grain_labels(ax, use_white_bg=False):
        for gid, (cx, cy) in grain_centroids.items():
            grain_size = np.sum(labels == gid)
            if grain_size > 10:
                if use_white_bg:
                    ax.text(cx, cy, str(gid), fontsize=8, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='black', alpha=0.8, linewidth=0.5))
                else:
                    ax.text(cx, cy, str(gid), fontsize=7, fontweight='bold',
                           ha='center', va='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                                    alpha=0.6, linewidth=0.3))
    
    # Panel 1: Grain Behavior Classification
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(behavior_map, cmap='RdYlGn', vmin=0, vmax=2, origin='lower', interpolation='nearest')
    ax1.set_title('Grain Behavior\n(Red=Plastic, Green=Elastic)', fontsize=10, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    add_grain_labels(ax1, use_white_bg=True)
    
    # Align predicted stress map if available
    pred_stress_map_aligned = pred_stress_map
    if pred_stress_map is not None:
        pred_stress_map_aligned, transform_pred = _auto_align_field_to_labels(pred_stress_map, labels)
        if transform_pred != 'identity':
            print(f"  Aligned predicted stress using transform: {transform_pred}")
    
    # Panel 2: GT Von Mises Stress (use viridis like E)
    ax_vm_gt = plt.subplot(3, 4, 2)
    if vm_stress_map_gt is not None:
        im_vm_gt = ax_vm_gt.imshow(vm_stress_map_gt, cmap='viridis',  # Same colormap as E
                                   vmin=stress_vmin, vmax=stress_vmax,
                                   origin='lower', interpolation='nearest')
        ax_vm_gt.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
        ax_vm_gt.set_title('σ_vM GT (MPa)', fontsize=10, fontweight='bold')
        ax_vm_gt.axis('off')
        plt.colorbar(im_vm_gt, ax=ax_vm_gt, fraction=0.046, pad=0.04)
        add_grain_labels(ax_vm_gt)
    else:
        ax_vm_gt.text(0.5, 0.5, 'GT Stress\n(Not Available)', 
                     ha='center', va='center', transform=ax_vm_gt.transAxes,
                     fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax_vm_gt.set_title('σ_vM GT (MPa)', fontsize=10, fontweight='bold')
        ax_vm_gt.axis('off')
    
    # Panel 3: Predicted Von Mises Stress (use viridis like E, same scale as GT)
    ax_vm_pred = plt.subplot(3, 4, 3)
    if pred_stress_map_aligned is not None:
        im_vm_pred = ax_vm_pred.imshow(pred_stress_map_aligned, cmap='viridis',  # Same colormap as E
                                       vmin=stress_vmin, vmax=stress_vmax,  # Same scale as GT
                                       origin='lower', interpolation='nearest')
        ax_vm_pred.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
        ax_vm_pred.set_title('σ_vM Predicted (MPa)', fontsize=10, fontweight='bold')
        ax_vm_pred.axis('off')
        plt.colorbar(im_vm_pred, ax=ax_vm_pred, fraction=0.046, pad=0.04)
        add_grain_labels(ax_vm_pred)
    else:
        ax_vm_pred.text(0.5, 0.5, 'Predicted Stress\n(Not Available)', 
                       ha='center', va='center', transform=ax_vm_pred.transAxes,
                       fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax_vm_pred.set_title('σ_vM Predicted (MPa)', fontsize=10, fontweight='bold')
        ax_vm_pred.axis('off')
    
    # Panel 4: Error Map (if both GT and predicted available)
    ax_err = plt.subplot(3, 4, 4)
    if vm_stress_map_gt is not None and pred_stress_map_aligned is not None:
        err_map = np.abs(vm_stress_map_gt - pred_stress_map_aligned)
        im_err = ax_err.imshow(err_map, cmap='viridis', origin='lower', interpolation='nearest')  # Use viridis
        ax_err.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
        ax_err.set_title('|GT - Pred| (MPa)', fontsize=10, fontweight='bold')
        ax_err.axis('off')
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
        add_grain_labels(ax_err)
    else:
        ax_err.text(0.5, 0.5, 'Error Map\n(Not Available)', 
                   ha='center', va='center', transform=ax_err.transAxes,
                   fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax_err.set_title('|GT - Pred| (MPa)', fontsize=10, fontweight='bold')
        ax_err.axis('off')
    
    # Property maps (Row 2)
    ax2 = plt.subplot(3, 4, 5)
    im2 = ax2.imshow(E_map, cmap='viridis', vmin=E_vmin, vmax=E_vmax,
                     origin='lower', interpolation='nearest')
    ax2.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
    ax2.set_title('Elastic Modulus E (GPa)', fontsize=10, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    add_grain_labels(ax2)
    
    ax3 = plt.subplot(3, 4, 6)
    im3 = ax3.imshow(nu_map, cmap='plasma', vmin=nu_vmin, vmax=nu_vmax,
                     origin='lower', interpolation='nearest')
    ax3.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
    ax3.set_title('Poisson\'s Ratio ν', fontsize=10, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    add_grain_labels(ax3)
    
    ax4 = plt.subplot(3, 4, 7)
    im4 = ax4.imshow(xi0_map, cmap='hot', vmin=xi0_vmin, vmax=xi0_vmax,
                     origin='lower', interpolation='nearest')
    ax4.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
    ax4.set_title('Initial Flow Stress ξ₀ (MPa)', fontsize=10, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    add_grain_labels(ax4)
    
    ax5 = plt.subplot(3, 4, 8)
    im5 = ax5.imshow(h0_map, cmap='coolwarm', vmin=h0_vmin, vmax=h0_vmax,
                     origin='lower', interpolation='nearest')
    ax5.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5, alpha=0.5)
    ax5.set_title('Hardening Modulus h₀ (GPa)', fontsize=10, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    add_grain_labels(ax5)
    
    # Statistics (Row 3)
    ax6 = plt.subplot(3, 4, 9)
    elastic_count = sum(1 for b in grain_behaviors.values() if b == 'elastic')
    plastic_count = sum(1 for b in grain_behaviors.values() if b == 'plastic')
    unknown_count = sum(1 for b in grain_behaviors.values() if b == 'unknown')
    
    ax6.bar(['Elastic', 'Plastic', 'Unknown'], 
           [elastic_count, plastic_count, unknown_count],
           color=['green', 'red', 'gray'], alpha=0.7)
    ax6.set_ylabel('Number of Grains', fontsize=10, fontweight='bold')
    ax6.set_title('Grain Behavior Distribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Property distributions for plastic vs elastic
    ax7 = plt.subplot(3, 4, 10)
    elastic_xi0 = [props[int(gid), 2]/1e6 for gid in unique_grains 
                   if grain_behaviors.get(int(gid), 'unknown') == 'elastic' and int(gid) < props.shape[0]]
    plastic_xi0 = [props[int(gid), 2]/1e6 for gid in unique_grains 
                   if grain_behaviors.get(int(gid), 'unknown') == 'plastic' and int(gid) < props.shape[0]]
    
    if len(elastic_xi0) > 0 and len(plastic_xi0) > 0:
        ax7.hist([elastic_xi0, plastic_xi0], bins=15, label=['Elastic', 'Plastic'], 
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax7.set_xlabel('Initial Flow Stress ξ₀ (MPa)', fontsize=10, fontweight='bold')
        ax7.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax7.set_title('ξ₀ Distribution by Behavior', fontsize=11, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    ax8 = plt.subplot(3, 4, 11)
    elastic_E = [props[int(gid), 0]/1e9 for gid in unique_grains 
                 if grain_behaviors.get(int(gid), 'unknown') == 'elastic' and int(gid) < props.shape[0]]
    plastic_E = [props[int(gid), 0]/1e9 for gid in unique_grains 
                 if grain_behaviors.get(int(gid), 'unknown') == 'plastic' and int(gid) < props.shape[0]]
    
    if len(elastic_E) > 0 and len(plastic_E) > 0:
        ax8.hist([elastic_E, plastic_E], bins=15, label=['Elastic', 'Plastic'], 
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax8.set_xlabel('Elastic Modulus E (GPa)', fontsize=10, fontweight='bold')
        ax8.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax8.set_title('E Distribution by Behavior', fontsize=11, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    
    # Summary text in title
    plastic_fraction = plastic_count/(elastic_count+plastic_count)*100 if (elastic_count+plastic_count) > 0 else 0
    plt.suptitle(f'Microstructure Analysis: Seed {seed} | Total: {len(unique_grains)} grains | '
                f'Elastic: {elastic_count} | Plastic: {plastic_count} ({plastic_fraction:.1f}%)', 
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_plastic_vs_elastic_prediction_quality(
    grain_results: Dict[int, Dict],
    grain_behaviors: Dict[int, str],
    seed: str,
    out_path: str
):
    """Plot comparison of prediction quality for plastic vs elastic grains."""
    elastic_mae = []
    plastic_mae = []
    elastic_rmse = []
    plastic_rmse = []
    
    for gid, results in grain_results.items():
        behavior = grain_behaviors.get(gid, 'unknown')
        if behavior == 'elastic' and 'mae' in results and not np.isnan(results['mae']):
            elastic_mae.append(results['mae'])
            elastic_rmse.append(results['rmse'])
        elif behavior == 'plastic' and 'mae' in results and not np.isnan(results['mae']):
            plastic_mae.append(results['mae'])
            plastic_rmse.append(results['rmse'])
    
    if len(elastic_mae) == 0 and len(plastic_mae) == 0:
        print(f"Warning: No valid data for plastic vs elastic comparison for seed {seed}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE comparison
    ax1 = axes[0]
    if len(elastic_mae) > 0 and len(plastic_mae) > 0:
        bp1 = ax1.boxplot([elastic_mae, plastic_mae], labels=['Elastic', 'Plastic'],
                          patch_artist=True, widths=0.6)
        bp1['boxes'][0].set_facecolor('green')
        bp1['boxes'][1].set_facecolor('red')
        for patch in bp1['boxes']:
            patch.set_alpha(0.7)
        
        # Add mean markers
        ax1.plot([1], [np.mean(elastic_mae)], 'o', color='darkgreen', markersize=10, label='Mean')
        ax1.plot([2], [np.mean(plastic_mae)], 'o', color='darkred', markersize=10)
        
        ax1.set_ylabel('MAE (MPa)', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Error (MAE) by Grain Behavior', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Add statistics text
        stats_text = f'Elastic: μ={np.mean(elastic_mae):.1f} MPa\nPlastic: μ={np.mean(plastic_mae):.1f} MPa'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    # RMSE comparison
    ax2 = axes[1]
    if len(elastic_rmse) > 0 and len(plastic_rmse) > 0:
        bp2 = ax2.boxplot([elastic_rmse, plastic_rmse], labels=['Elastic', 'Plastic'],
                          patch_artist=True, widths=0.6)
        bp2['boxes'][0].set_facecolor('green')
        bp2['boxes'][1].set_facecolor('red')
        for patch in bp2['boxes']:
            patch.set_alpha(0.7)
        
        # Add mean markers
        ax2.plot([1], [np.mean(elastic_rmse)], 'o', color='darkgreen', markersize=10, label='Mean')
        ax2.plot([2], [np.mean(plastic_rmse)], 'o', color='darkred', markersize=10)
        
        ax2.set_ylabel('RMSE (MPa)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Error (RMSE) by Grain Behavior', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Add statistics text
        stats_text = f'Elastic: μ={np.mean(elastic_rmse):.1f} MPa\nPlastic: μ={np.mean(plastic_rmse):.1f} MPa'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.suptitle(f'ML Prediction Quality: Plastic vs Elastic Grains (Seed {seed})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ML_DATASET', help='ML dataset root directory')
    ap.add_argument('--predictions', default='ML_EVAL/predictions', help='Predictions directory')
    ap.add_argument('--hdf5-dir', default='simulation_results/hdf5_files', help='HDF5 files directory')
    ap.add_argument('--labels-dir', default='labels', help='Labels directory')
    ap.add_argument('--props-dir', default='props', help='Properties directory')
    ap.add_argument('--out', default='ML_EVAL/per_grain_test_analysis', help='Output directory')
    ap.add_argument('--metadata-csv', default='ML_DATASET/metadata/increments_map.csv', help='Metadata CSV')
    ap.add_argument('--seeds', default=None, help='Comma-separated list of seeds to analyze (default: all test seeds)')
    ap.add_argument('--max-grains', type=int, default=20, help='Maximum grains to plot per seed')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(args.metadata_csv)
    
    # Get test seeds
    if args.seeds:
        test_seeds = [s.strip() for s in args.seeds.split(',')]
    else:
        # Get all test seeds from metadata
        test_seeds = list(set([info['seed'] for info in metadata.values() if info['split'] == 'test']))
        test_seeds.sort()
    
    print(f"Analyzing {len(test_seeds)} test seeds: {test_seeds}")
    
    # Process each seed
    all_results = {}
    
    for seed in test_seeds:
        print(f"\n{'='*60}")
        print(f"Processing seed {seed}")
        print(f"{'='*60}")
        
        # Find labels and props
        labels_path = find_labels_path(seed, args.labels_dir)
        if labels_path is None:
            print(f"Warning: Labels not found for seed {seed}, skipping")
            continue
        
        props_path = find_props_path(seed, args.props_dir)
        if props_path is None:
            print(f"Warning: Props not found for seed {seed}, skipping")
            print(f"  Searched in: {args.props_dir}")
            print(f"  Looking for: props_seed{seed.replace('seed', '')}.npy or similar")
            continue
        
        # Handle seeds with or without "seed" prefix
        seed_clean = seed.replace('seed', '') if seed.startswith('seed') else seed
        h5_path = os.path.join(args.hdf5_dir, f"seed{seed_clean}.hdf5")
        if not os.path.exists(h5_path):
            # Try alternative naming
            alt_path = os.path.join(args.hdf5_dir, f"{seed}.hdf5")
            if os.path.exists(alt_path):
                h5_path = alt_path
            else:
                print(f"Warning: HDF5 not found for seed {seed}, skipping")
                print(f"  Searched: {h5_path}")
                print(f"  Also tried: {alt_path}")
                continue
        
        # Load labels and props
        labels = np.load(labels_path)
        if labels.ndim == 0:
            labels = labels.item()
        
        props_raw = np.load(props_path, allow_pickle=True)
        if props_raw.ndim == 0 and isinstance(props_raw.item(), dict):
            d = props_raw.item()
            props = np.stack([
                np.asarray(d['E']),
                np.asarray(d['nu']),
                np.asarray(d['xi0']),
                np.asarray(d['h0'])
            ], axis=1)
        else:
            props = props_raw
        
        unique_grains = np.unique(labels)
        print(f"Found {len(unique_grains)} grains")
        
        # Load GT curves
        print("Loading ground truth curves from HDF5...")
        gt_grain_data = load_gt_per_grain_curves(h5_path, labels)
        
        # Find a test sample for this seed
        test_samples = [sname for sname, info in metadata.items() 
                       if info['seed'] == seed and info['split'] == 'test']
        if len(test_samples) == 0:
            print(f"Warning: No test samples found for seed {seed}, skipping")
            continue
        
        # Use first test sample to get predictions
        sample_name = test_samples[0]
        print(f"Loading predicted curves from ML predictions (using {sample_name})...")
        pred_grain_data = load_predicted_per_grain_curves(
            sample_name, args.predictions, args.data, labels, metadata, h5_path
        )
        
        # Create output directory for this seed (handle seed prefix)
        seed_clean = seed.replace('seed', '') if seed.startswith('seed') else seed
        seed_out_dir = os.path.join(args.out, f"seed{seed_clean}")
        os.makedirs(seed_out_dir, exist_ok=True)
        
        # Classify grain behaviors
        print("Classifying grain behaviors (elastic vs plastic)...")
        grain_behaviors = {}
        for gid in unique_grains:
            gid = int(gid)
            if gid in gt_grain_data:
                behavior, deviation = classify_grain_behavior(gt_grain_data[gid], props, gid)
                grain_behaviors[gid] = behavior
        
        # Load predicted stress map for the last increment
        pred_stress_map = None
        if len(test_samples) > 0:
            # Get the last sample (highest increment) for this seed
            last_sample = None
            last_inc = -1
            for sname in test_samples:
                if metadata[sname]['increment_t'] > last_inc:
                    last_inc = metadata[sname]['increment_t']
                    last_sample = sname
            
            if last_sample:
                pred_path = os.path.join(args.predictions, f"{last_sample}.npy")
                if os.path.exists(pred_path):
                    pred = np.load(pred_path)
                    if pred.ndim == 3:
                        pred = pred[0]  # Remove channel dimension if present
                    # Denormalize: pred_mpa = pred * 1000.0
                    pred_stress_map = pred * 1000.0  # Convert to MPa
        
        # Generate microstructure with properties plot
        print("Generating microstructure analysis plot...")
        microstruct_path = os.path.join(seed_out_dir, 'microstructure_analysis.png')
        plot_microstructure_with_properties(labels, props, grain_behaviors, seed, microstruct_path, h5_path, pred_stress_map)
        
        # Plot each grain (limit to max_grains)
        grains_to_plot = sorted(unique_grains)[:args.max_grains]
        print(f"Plotting {len(grains_to_plot)} grains...")
        
        seed_results = {}
        
        for gid in grains_to_plot:
            gid = int(gid)
            if gid not in gt_grain_data or gid not in pred_grain_data:
                continue
            
            gt_data = gt_grain_data[gid]
            pred_data = pred_grain_data[gid]
            
            if len(gt_data['strain']) == 0 or len(pred_data['strain']) == 0:
                continue
            
            # Get behavior classification
            behavior = grain_behaviors.get(gid, 'unknown')
            
            # Plot
            out_path = os.path.join(seed_out_dir, f"grain_{gid:03d}_comparison.png")
            plot_per_grain_comparison(gid, gt_data, pred_data, props, out_path, behavior)
            
            # Compute error metrics
            # Interpolate to common strain points for comparison
            strain_min = max(gt_data['strain'].min(), pred_data['strain'].min())
            strain_max = min(gt_data['strain'].max(), pred_data['strain'].max())
            
            if strain_max > strain_min:
                strain_common = np.linspace(strain_min, strain_max, 50)
                
                if HAS_SCIPY:
                    try:
                        gt_interp = interp1d(gt_data['strain'], gt_data['stress_vm'], 
                                           kind='linear', fill_value='extrapolate')
                        pred_interp = interp1d(pred_data['strain'], pred_data['stress_vm'], 
                                             kind='linear', fill_value='extrapolate')
                        
                        gt_common = gt_interp(strain_common)
                        pred_common = pred_interp(strain_common)
                    except Exception as e:
                        # Fallback to simple nearest neighbor
                        gt_common = np.interp(strain_common, gt_data['strain'], gt_data['stress_vm'])
                        pred_common = np.interp(strain_common, pred_data['strain'], pred_data['stress_vm'])
                else:
                    # Use numpy interpolation
                    gt_common = np.interp(strain_common, gt_data['strain'], gt_data['stress_vm'])
                    pred_common = np.interp(strain_common, pred_data['strain'], pred_data['stress_vm'])
                
                mae = np.mean(np.abs(gt_common - pred_common))
                rmse = np.sqrt(np.mean((gt_common - pred_common)**2))
                
                seed_results[gid] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'n_points_gt': len(gt_data['strain']),
                    'n_points_pred': len(pred_data['strain']),
                    'behavior': grain_behaviors.get(gid, 'unknown')
                }
            else:
                seed_results[gid] = {
                    'mae': float('nan'),
                    'rmse': float('nan'),
                    'n_points_gt': len(gt_data['strain']),
                    'n_points_pred': len(pred_data['strain']),
                    'behavior': grain_behaviors.get(gid, 'unknown')
                }
        
        all_results[seed] = seed_results
        
        # Generate plastic vs elastic prediction quality comparison
        print("Generating plastic vs elastic prediction quality plot...")
        quality_path = os.path.join(seed_out_dir, 'plastic_vs_elastic_prediction_quality.png')
        plot_plastic_vs_elastic_prediction_quality(seed_results, grain_behaviors, seed, quality_path)
        
        # Save summary for this seed (include behavior info)
        summary_data = {
            'grain_results': seed_results,
            'grain_behaviors': grain_behaviors,
            'statistics': {
                'total_grains': len(unique_grains),
                'elastic_count': sum(1 for b in grain_behaviors.values() if b == 'elastic'),
                'plastic_count': sum(1 for b in grain_behaviors.values() if b == 'plastic'),
                'unknown_count': sum(1 for b in grain_behaviors.values() if b == 'unknown'),
            }
        }
        
        # Add error statistics by behavior
        elastic_mae = [r['mae'] for r in seed_results.values() 
                      if r.get('behavior') == 'elastic' and not np.isnan(r.get('mae', np.nan))]
        plastic_mae = [r['mae'] for r in seed_results.values() 
                      if r.get('behavior') == 'plastic' and not np.isnan(r.get('mae', np.nan))]
        
        if len(elastic_mae) > 0:
            summary_data['statistics']['elastic_mae_mean'] = float(np.mean(elastic_mae))
            summary_data['statistics']['elastic_mae_std'] = float(np.std(elastic_mae))
        if len(plastic_mae) > 0:
            summary_data['statistics']['plastic_mae_mean'] = float(np.mean(plastic_mae))
            summary_data['statistics']['plastic_mae_std'] = float(np.std(plastic_mae))
        
        summary_path = os.path.join(seed_out_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Saved {len(seed_results)} grain comparisons to {seed_out_dir}")
    
    # Save overall summary
    overall_summary = {
        'seeds_analyzed': list(all_results.keys()),
        'per_seed_results': all_results
    }
    
    summary_path = os.path.join(args.out, 'overall_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {args.out}")
    print(f"Overall summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

