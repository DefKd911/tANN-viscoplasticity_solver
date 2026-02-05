#!/usr/bin/env python3
"""
Analyze per-grain stress-strain curves from DAMASK HDF5 output.
For each grain, extracts SS curve and classifies as plastic vs elastic based on deviation from elastic estimate.

Usage:
  python analyze_per_grain_ss_curves.py --seed 105447566 --hdf5-dir simulation_results/hdf5_files --labels-dir labels --props-dir props --out per_grain_analysis
"""
import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional
import json

# Reuse tensor loading logic from existing scripts
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
    """Reshape tensor to (N, 3, 3) format."""
    if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        return arr
    if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    if arr.ndim == 5 and arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    raise ValueError(f"Unexpected tensor shape {arr.shape}")


def _load_dataset(f: h5py.File, path: str) -> Optional[np.ndarray]:
    if path in f:
        return f[path][()]
    return None


def _load_tensor(f: h5py.File, base: str, rel_paths: List[str]) -> Optional[np.ndarray]:
    for rel in rel_paths:
        path = f"{base}/{rel}"
        data = _load_dataset(f, path)
        if data is not None:
            return _reshape_tensor(data)
    return None


def compute_cauchy_stress(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute Cauchy stress from F and P: sigma = (1/detF) * P * F^T"""
    if F.ndim != 3 or F.shape[-2:] != (3, 3):
        raise ValueError(f"F must be (N, 3, 3), got {F.shape}")
    N = F.shape[0]
    sigma = np.zeros_like(P)
    for i in range(N):
        F_i = F[i]
        P_i = P[i]
        det_F = np.linalg.det(F_i)
        if abs(det_F) > 1e-10:
            sigma[i] = (1.0 / det_F) * P_i @ F_i.T
        else:
            sigma[i] = np.eye(3) * 1e6  # fallback
    return sigma


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    """Compute von Mises stress from Cauchy stress tensor (N, 3, 3) -> (N,)"""
    if sigma.ndim != 3 or sigma.shape[-2:] != (3, 3):
        raise ValueError(f"sigma must be (N, 3, 3), got {sigma.shape}")
    N = sigma.shape[0]
    sigma_vM = np.zeros(N)
    for i in range(N):
        s = sigma[i]
        s_dev = s - np.trace(s) / 3.0 * np.eye(3)
        sigma_vM[i] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
    return sigma_vM


def extract_strain_from_F(F: np.ndarray) -> np.ndarray:
    """Extract engineering strain ε11 from F for each voxel: ε11 = F11 - 1"""
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


def load_per_grain_data(h5_path: str, labels: np.ndarray) -> Dict[int, Dict[str, List[float]]]:
    """
    Load stress-strain data per grain across all increments.
    Returns: {grain_id: {'strain': [...], 'stress_vm': [...], 'stress_11': [...]}}
    """
    grain_data = {}
    unique_grains = np.unique(labels)
    
    # Initialize dict for each grain
    for gid in unique_grains:
        grain_data[int(gid)] = {'strain': [], 'stress_vm': [], 'stress_11': []}
    
    H, W = labels.shape
    N_voxels = H * W
    
    with h5py.File(h5_path, 'r') as f:
        increments = _list_increment_groups(f)
        print(f"Found {len(increments)} increments")
        
        for inc_name in increments:
            hom_path = f'{inc_name}/homogenization/h0'
            
            if f"{hom_path}/mechanical" not in f:
                continue
            
            # Try to load sigma directly
            sigma = _load_tensor(f, hom_path, SIGMA_DATASETS)
            if sigma is None:
                # Compute from F and P
                F = _load_tensor(f, hom_path, F_DATASETS)
                P = _load_tensor(f, hom_path, P_DATASETS)
                if F is None or P is None:
                    continue
                sigma = compute_cauchy_stress(F, P)
            
            # Get F for strain
            F = _load_tensor(f, hom_path, F_DATASETS)
            if F is None:
                continue
            
            # Reshape to spatial grid if needed
            if sigma.shape[0] != N_voxels:
                # Assume it's already flattened or needs reshaping
                if sigma.shape[0] == H * W:
                    sigma_flat = sigma
                    F_flat = F
                else:
                    print(f"Warning: sigma shape {sigma.shape[0]} != {N_voxels}, skipping increment")
                    continue
            else:
                sigma_flat = sigma
                F_flat = F
            
            # Compute per-voxel quantities
            strain_11 = extract_strain_from_F(F_flat)  # (N_voxels,)
            stress_vm = von_mises_stress(sigma_flat)  # (N_voxels,)
            stress_11 = sigma_flat[:, 0, 0]  # (N_voxels,)
            
            # Average per grain
            labels_flat = labels.flatten()
            for gid in unique_grains:
                mask = (labels_flat == gid)
                if mask.sum() > 0:
                    grain_data[int(gid)]['strain'].append(strain_11[mask].mean() * 100)  # convert to %
                    grain_data[int(gid)]['stress_vm'].append(stress_vm[mask].mean() / 1e6)  # convert to MPa
                    grain_data[int(gid)]['stress_11'].append(stress_11[mask].mean() / 1e6)  # convert to MPa
    
    return grain_data


def compute_elastic_estimate(E: float, nu: float, strain: np.ndarray) -> np.ndarray:
    """
    Compute elastic von Mises stress estimate for uniaxial tension.
    For uniaxial tension: σ11 = E * ε11, and von Mises stress = σ11.
    So: σ_vM_el = E * ε11
    """
    return (E * strain) / 1e6  # Convert to MPa


def detect_yield_point(
    strain: np.ndarray,
    stress_vm: np.ndarray,
    stress_el: np.ndarray,
    xi0_MPa: float,
    E_GPa: float,
    yield_threshold_ratio: float = 0.02
) -> Optional[int]:
    """
    Detect yield point using the actual yield stress (xi0) for each grain.
    
    Each grain has different properties, so they should yield at different strains:
    - Expected yield strain: ε_yield ≈ xi0 / E
    - Yield occurs when: stress_vm >= xi0
    
    Method: Find where stress_vm first reaches or exceeds xi0 (with small tolerance).
    This gives grain-specific yield points, not a fixed strain value.
    
    Returns index of yield point, or None if no clear yield.
    """
    if len(strain) < 3:
        return None
    
    # Expected yield strain based on material properties
    # ε_yield = xi0 / E (for uniaxial tension)
    E_MPa = E_GPa * 1000  # Convert GPa to MPa
    expected_yield_strain = (xi0_MPa / E_MPa) * 100  # Convert to %
    
    # Method 1: Find where stress first reaches or exceeds yield stress (xi0)
    # Use a tolerance (95% of xi0) to account for numerical precision
    yield_stress_threshold = xi0_MPa * 0.95
    yield_by_stress = np.where(stress_vm >= yield_stress_threshold)[0]
    
    if len(yield_by_stress) > 0:
        # Return the FIRST point where stress reaches yield
        return int(yield_by_stress[0])
    
    # Method 2: If stress never reaches xi0, check for significant deviation from elastic
    # This handles cases where yield stress is very high
    deviation = np.abs((stress_vm - stress_el) / (stress_el + 1e-6))
    significant_deviation = np.where(deviation > 0.05)[0]  # 5% deviation
    
    if len(significant_deviation) > 0:
        # Use the first significant deviation as yield point
        return int(significant_deviation[0])
    
    # Method 3: If expected yield strain is within range, find closest point
    if expected_yield_strain > 0 and expected_yield_strain < strain.max():
        closest_idx = np.argmin(np.abs(strain - expected_yield_strain))
        # Only return if stress is reasonably close to xi0
        if stress_vm[closest_idx] >= xi0_MPa * 0.8:
            return int(closest_idx)
    
    return None


def classify_grain_plasticity(
    strain: np.ndarray,
    stress_vm: np.ndarray,
    E: float,
    nu: float,
    threshold_ratio: float = 0.1
) -> Tuple[str, float]:
    """
    Classify grain as 'plastic' or 'elastic' based on deviation from elastic line.
    Returns: ('plastic' or 'elastic', mean_deviation_ratio)
    """
    if len(strain) < 2:
        return 'elastic', 0.0
    
    # Convert strain from % back to fraction
    strain_frac = strain / 100.0
    
    # Elastic estimate
    stress_el = compute_elastic_estimate(E, nu, strain_frac)
    
    # Deviation ratio: (actual - elastic) / elastic
    deviation_ratio = (stress_vm - stress_el) / (stress_el + 1e-6)  # avoid div by zero
    
    # Use mean absolute deviation ratio
    mean_dev = np.abs(deviation_ratio).mean()
    
    if mean_dev > threshold_ratio:
        return 'plastic', mean_dev
    else:
        return 'elastic', mean_dev


def load_props(labels: np.ndarray, props_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load properties and map to per-grain arrays."""
    props_raw = np.load(props_path, allow_pickle=True)
    
    # Handle dict format
    if props_raw.ndim == 0 and isinstance(props_raw.item(), dict):
        d = props_raw.item()
        E_arr = np.asarray(d['E'])
        nu_arr = np.asarray(d['nu'])
        xi0_arr = np.asarray(d['xi0'])
        h0_arr = np.asarray(d['h0'])
        props = np.stack([E_arr, nu_arr, xi0_arr, h0_arr], axis=1)
    else:
        props = props_raw
    
    # Map to labels
    max_label = labels.max()
    if max_label >= props.shape[0]:
        raise ValueError(f"labels contain grain id {max_label} but props only has {props.shape[0]} rows")
    
    E_map = props[labels, 0]
    nu_map = props[labels, 1]
    xi0_map = props[labels, 2]
    h0_map = props[labels, 3]
    
    return E_map, nu_map, xi0_map, h0_map


def main():
    ap = argparse.ArgumentParser(description='Analyze per-grain stress-strain curves')
    ap.add_argument('--seed', type=str, required=True, help='Seed ID (e.g., 105447566)')
    ap.add_argument('--hdf5-dir', type=str, default='simulation_results/hdf5_files', help='Directory with HDF5 files')
    ap.add_argument('--labels-dir', type=str, default='labels', help='Directory with label .npy files')
    ap.add_argument('--props-dir', type=str, default='props', help='Directory with props .npy files')
    ap.add_argument('--out', type=str, default='per_grain_analysis', help='Output directory')
    ap.add_argument('--threshold', type=float, default=0.1, help='Threshold ratio for plastic classification (default: 0.1)')
    args = ap.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    seed = args.seed
    h5_path = os.path.join(args.hdf5_dir, f'seed{seed}.hdf5')
    
    # Try multiple label file naming conventions
    labels_path = None
    for label_name in [f'labels_seed{seed}.npy', f'seed{seed}.npy']:
        candidate = os.path.join(args.labels_dir, label_name)
        if os.path.isfile(candidate):
            labels_path = candidate
            break
    
    # Try multiple props file naming conventions
    props_path = None
    for prop_name in [f'props_seed{seed}.npy', f'seed{seed}.npy']:
        candidate = os.path.join(args.props_dir, prop_name)
        if os.path.isfile(candidate):
            props_path = candidate
            break
    
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    if labels_path is None:
        raise FileNotFoundError(f"Labels file not found in {args.labels_dir} for seed {seed}")
    if props_path is None:
        raise FileNotFoundError(f"Props file not found in {args.props_dir} for seed {seed}")
    
    print(f"Loading data for seed {seed}...")
    labels = np.load(labels_path)
    if labels.ndim != 2:
        raise ValueError(f"Labels must be 2D, got shape {labels.shape}")
    
    E_map, nu_map, xi0_map, h0_map = load_props(labels, props_path)
    
    print(f"Extracting per-grain stress-strain data...")
    grain_data = load_per_grain_data(h5_path, labels)
    
    unique_grains = sorted(grain_data.keys())
    print(f"Found {len(unique_grains)} grains")
    
    # Classify grains
    grain_classifications = {}
    for gid in unique_grains:
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        
        if len(strain) == 0:
            continue
        
        # Get grain properties (average over grain pixels)
        mask = (labels == gid)
        E_grain = E_map[mask].mean()
        nu_grain = nu_map[mask].mean()
        xi0_grain = xi0_map[mask].mean()
        h0_grain = h0_map[mask].mean()
        
        classification, dev_ratio = classify_grain_plasticity(
            strain, stress_vm, E_grain, nu_grain, threshold_ratio=args.threshold
        )
        
        grain_classifications[gid] = {
            'classification': classification,
            'deviation_ratio': float(dev_ratio),
            'E': float(E_grain),
            'nu': float(nu_grain),
            'xi0': float(xi0_grain),
            'h0': float(h0_grain),
            'num_points': len(strain)
        }
    
    # Separate plastic and elastic grains
    plastic_grains = [gid for gid, info in grain_classifications.items() if info['classification'] == 'plastic']
    elastic_grains = [gid for gid, info in grain_classifications.items() if info['classification'] == 'elastic']
    
    print(f"\nClassification Results:")
    print(f"  Plastic grains: {len(plastic_grains)} ({len(plastic_grains)/len(unique_grains)*100:.1f}%)")
    print(f"  Elastic grains: {len(elastic_grains)} ({len(elastic_grains)/len(unique_grains)*100:.1f}%)")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: All grains SS curves (colored by classification)
    ax1 = plt.subplot(2, 3, 1)
    for gid in unique_grains:
        if len(grain_data[gid]['strain']) == 0:
            continue
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        classification = grain_classifications[gid]['classification']
        color = 'red' if classification == 'plastic' else 'blue'
        alpha = 0.6 if classification == 'plastic' else 0.3
        ax1.plot(strain, stress_vm, color=color, alpha=alpha, linewidth=1.5)
    
    ax1.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Von Mises Stress (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title(f'All Grains SS Curves (Red=Plastic, Blue=Elastic)\nSeed {seed}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Plastic', 'Elastic'], loc='upper left')
    
    # Plot 2: Plastic grains only
    ax2 = plt.subplot(2, 3, 2)
    for gid in plastic_grains:
        if len(grain_data[gid]['strain']) == 0:
            continue
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        # Also plot elastic estimate
        E_g = grain_classifications[gid]['E']
        nu_g = grain_classifications[gid]['nu']
        strain_frac = strain / 100.0
        stress_el = compute_elastic_estimate(E_g, nu_g, strain_frac)
        ax2.plot(strain, stress_vm, 'r-', linewidth=2, alpha=0.7, label=f'Grain {gid}' if gid == plastic_grains[0] else '')
        ax2.plot(strain, stress_el, 'r--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Von Mises Stress (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Plastic Grains ({len(plastic_grains)} grains)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(plastic_grains) > 0:
        ax2.legend(['Actual', 'Elastic Est.'], loc='upper left', fontsize=9)
    
    # Plot 3: Elastic grains only
    ax3 = plt.subplot(2, 3, 3)
    for gid in elastic_grains:
        if len(grain_data[gid]['strain']) == 0:
            continue
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        E_g = grain_classifications[gid]['E']
        nu_g = grain_classifications[gid]['nu']
        strain_frac = strain / 100.0
        stress_el = compute_elastic_estimate(E_g, nu_g, strain_frac)
        ax3.plot(strain, stress_vm, 'b-', linewidth=2, alpha=0.7)
        ax3.plot(strain, stress_el, 'b--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Von Mises Stress (MPa)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Elastic Grains ({len(elastic_grains)} grains)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    if len(elastic_grains) > 0:
        ax3.legend(['Actual', 'Elastic Est.'], loc='upper left', fontsize=9)
    
    # Plot 4: Classification map
    ax4 = plt.subplot(2, 3, 4)
    classification_map = np.zeros_like(labels, dtype=float)
    for gid in unique_grains:
        if gid in grain_classifications:
            classification_map[labels == gid] = 1.0 if grain_classifications[gid]['classification'] == 'plastic' else 0.0
    im4 = ax4.imshow(classification_map, cmap='RdYlBu', vmin=0, vmax=1, origin='lower', interpolation='nearest')
    ax4.set_title('Classification Map (Red=Plastic, Blue=Elastic)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('X (pixels)', fontsize=11)
    ax4.set_ylabel('Y (pixels)', fontsize=11)
    plt.colorbar(im4, ax=ax4, label='Plastic (1.0) / Elastic (0.0)')
    
    # Plot 5: Deviation ratio map
    ax5 = plt.subplot(2, 3, 5)
    deviation_map = np.zeros_like(labels, dtype=float)
    for gid in unique_grains:
        if gid in grain_classifications:
            deviation_map[labels == gid] = grain_classifications[gid]['deviation_ratio']
    im5 = ax5.imshow(deviation_map, cmap='hot', origin='lower', interpolation='nearest')
    ax5.set_title('Mean Deviation Ratio Map', fontsize=13, fontweight='bold')
    ax5.set_xlabel('X (pixels)', fontsize=11)
    ax5.set_ylabel('Y (pixels)', fontsize=11)
    plt.colorbar(im5, ax=ax5, label='|(σ_actual - σ_el)| / σ_el')
    
    # Plot 6: Property comparison (E, xi0) colored by classification
    ax6 = plt.subplot(2, 3, 6)
    E_vals = [grain_classifications[gid]['E'] / 1e9 for gid in unique_grains if gid in grain_classifications]
    xi0_vals = [grain_classifications[gid]['xi0'] / 1e6 for gid in unique_grains if gid in grain_classifications]
    colors = ['red' if grain_classifications[gid]['classification'] == 'plastic' else 'blue' 
              for gid in unique_grains if gid in grain_classifications]
    ax6.scatter(E_vals, xi0_vals, c=colors, alpha=0.6, s=50)
    ax6.set_xlabel('E (GPa)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('xi0 (MPa)', fontsize=12, fontweight='bold')
    ax6.set_title('Property Space (Red=Plastic, Blue=Elastic)', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(['Plastic', 'Elastic'], loc='upper left')
    
    plt.tight_layout()
    
    # Save main figure
    out_file = os.path.join(args.out, f'seed{seed}_per_grain_analysis.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Main analysis: {out_file}")
    
    # Save individual grain SS curves - show ALL grains
    # Sort all grains: plastic first (by deviation desc), then elastic (by deviation asc)
    all_grains_sorted = (
        sorted(plastic_grains, key=lambda g: grain_classifications[g]['deviation_ratio'], reverse=True) +
        sorted(elastic_grains, key=lambda g: grain_classifications[g]['deviation_ratio'])
    )
    
    total_grains = len(all_grains_sorted)
    if total_grains == 0:
        print("No grains to plot!")
        return
    
    # Calculate grid size: try to make it roughly square
    num_cols = int(np.ceil(np.sqrt(total_grains * 1.5)))  # Slightly wider than tall
    num_rows = int(np.ceil(total_grains / num_cols))
    
    fig2, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows))
    if total_grains == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot all grains
    for idx, gid in enumerate(all_grains_sorted):
        ax = axes[idx]
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        E_g = grain_classifications[gid]['E']
        nu_g = grain_classifications[gid]['nu']
        classification = grain_classifications[gid]['classification']
        dev_ratio = grain_classifications[gid]['deviation_ratio']
        
        strain_frac = strain / 100.0
        stress_el = compute_elastic_estimate(E_g, nu_g, strain_frac)
        
        # Get grain properties for yield detection (use stored xi0 and E from classification)
        xi0_grain = grain_classifications[gid]['xi0'] / 1e6  # Convert to MPa
        E_grain_GPa = grain_classifications[gid]['E'] / 1e9  # Convert to GPa
        
        # Detect yield point using actual yield stress (xi0) and E
        yield_idx = detect_yield_point(strain, stress_vm, stress_el, xi0_grain, E_grain_GPa, yield_threshold_ratio=0.02)
        
        # Color by classification
        color = 'red' if classification == 'plastic' else 'blue'
        
        # Plot elastic estimate (dashed gray line) - THEORETICAL reference only
        # This is ideal Hooke's law: σ = E×ε (assumes pure elasticity, no microstructure)
        # NOTE: DAMASK simulation will NEVER perfectly match this because:
        # - Heterogeneous stiffness (grain-to-grain variation)
        # - Elastic anisotropy (crystal orientation effects)
        # - Voxel averaging (spatial discretization)
        # - Micro-plasticity at very low strain
        # - Grain boundary interactions
        # - Rate-dependent viscoplasticity (not pure elasticity)
        ax.plot(strain, stress_el, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
               label='Theoretical Elastic (E×ε, ideal reference)', zorder=1)
        
        # For plastic grains, plot elastic and hardening regions separately
        if classification == 'plastic' and yield_idx is not None and yield_idx > 0:
            # Plot "elastic region" (before detected yield)
            # IMPORTANT: This is ACTUAL DAMASK simulation data, NOT pure elastic response!
            # It includes microstructure effects, anisotropy, micro-plasticity, etc.
            # It will NOT perfectly match the gray theoretical line - this is expected!
            if yield_idx < len(strain):
                # Calculate deviation to show how much microstructure affects the response
                elastic_deviation = np.abs((stress_vm[:yield_idx+1] - stress_el[:yield_idx+1]) / (stress_el[:yield_idx+1] + 1e-6))
                max_dev = elastic_deviation.max() if len(elastic_deviation) > 0 else 0
                
                # Plot actual DAMASK response before yield (includes microstructure effects)
                ax.plot(strain[:yield_idx+1], stress_vm[:yield_idx+1], 
                       color='green', linewidth=3, alpha=0.9, 
                       label=f'Pre-Yield Response (GT, dev={max_dev*100:.1f}%)', zorder=3)
            # Plot hardening region (after yield) - actual DAMASK data
            if yield_idx < len(strain) - 1:
                ax.plot(strain[yield_idx:], stress_vm[yield_idx:], 
                       color='red', linewidth=3, alpha=0.9, label='Hardening Region (GT)', zorder=4)
        else:
            # For elastic grains or if no yield detected, plot full curve
            # These grains stay close to elastic behavior throughout (low deviation)
            ax.plot(strain, stress_vm, color=color, linewidth=2.5, label='Actual Response (GT)', zorder=3)
        
        # Mark yield point if detected
        if yield_idx is not None and yield_idx < len(strain):
            ax.plot(strain[yield_idx], stress_vm[yield_idx], 'ko', markersize=10, 
                   label='Yield Point', zorder=5, markerfacecolor='yellow', markeredgewidth=2)
            # Draw vertical line at yield
            ax.axvline(strain[yield_idx], color='orange', linestyle=':', linewidth=2, alpha=0.8, zorder=2)
        
        title = f'Grain {gid} ({classification.capitalize()}, dev={dev_ratio:.3f})'
        if yield_idx is not None:
            title += f'\nYield at {strain[yield_idx]:.3f}%'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Strain (%)', fontsize=9)
        ax.set_ylabel('σvM (MPa)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')
    
    # Hide unused subplots
    for idx in range(total_grains, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    out_file2 = os.path.join(args.out, f'seed{seed}_individual_grains.png')
    plt.savefig(out_file2, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Individual grains: {out_file2}")
    
    # Save summary JSON
    summary = {
        'seed': seed,
        'total_grains': len(unique_grains),
        'plastic_count': len(plastic_grains),
        'elastic_count': len(elastic_grains),
        'plastic_percentage': len(plastic_grains) / len(unique_grains) * 100,
        'elastic_percentage': len(elastic_grains) / len(unique_grains) * 100,
        'threshold_ratio': args.threshold,
        'grain_classifications': grain_classifications
    }
    
    json_file = os.path.join(args.out, f'seed{seed}_summary.json')
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Summary JSON: {json_file}")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Per-grain analysis complete for seed {seed}!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

