#!/usr/bin/env python3
"""
Enhanced per-grain stress-strain analysis with:
1. Real elastic estimation using C11/C12/C44 (cubic crystal elasticity)
2. Comparison with isotropic E×ε estimate
3. Better yield detection based on actual material properties

Usage:
  python analyze_per_grain_ss_curves_enhanced.py --seed 105447566 --hdf5-dir simulation_results/hdf5_files --labels-dir labels --props-dir props --material-dir material_yaml_fixed --out per_grain_analysis
"""
import os
import argparse
import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

# Reuse tensor loading logic
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


def compute_cubic_elastic_stress(C11: float, C12: float, C44: float, strain: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stress for cubic crystal under uniaxial tension.
    
    For cubic crystal under uniaxial tension in [100] direction:
    - σ11 = (C11^2 + C11*C12 - C12^2)/(C11 + C12) * ε11
    - For von Mises in uniaxial tension: σ_vM = σ11
    
    This is more accurate than isotropic E×ε for cubic crystals.
    """
    # Effective Young's modulus for [100] direction in cubic crystal
    # E[100] = (C11 - C12)(C11 + 2*C12) / (C11 + C12)
    # But for stress: σ11 = (C11^2 + C11*C12 - C12^2)/(C11 + C12) * ε11
    numerator = C11 * C11 + C11 * C12 - C12 * C12
    denominator = C11 + C12
    if abs(denominator) < 1e-10:
        # Fallback to isotropic
        E_eff = (C11 - C12) * (C11 + 2*C12) / (C11 + C12 + 1e-10)
    else:
        E_eff = numerator / denominator
    
    # Von Mises stress = σ11 for uniaxial tension
    sigma_vm = E_eff * strain
    return sigma_vm / 1e6  # Convert to MPa


def compute_isotropic_elastic_stress(E: float, nu: float, strain: np.ndarray) -> np.ndarray:
    """
    Compute isotropic elastic stress estimate: σ_vM = E × ε
    """
    return (E * strain) / 1e6  # Convert to MPa


def load_elastic_constants(material_yaml_path: str, grain_id: int) -> Tuple[float, float, float]:
    """
    Load C11, C12, C44 for a specific grain from material YAML.
    Returns (C11, C12, C44) in Pa.
    """
    try:
        with open(material_yaml_path, 'r') as f:
            mat = yaml.safe_load(f)
        
        grain_key = f'grain_{grain_id}'
        if 'phase' in mat and grain_key in mat['phase']:
            phase = mat['phase'][grain_key]
            if 'mechanical' in phase and 'elastic' in phase['mechanical']:
                elastic = phase['mechanical']['elastic']
                C11 = float(elastic.get('C_11', 0))
                C12 = float(elastic.get('C_12', 0))
                C44 = float(elastic.get('C_44', 0))
                return C11, C12, C44
    except Exception as e:
        print(f"Warning: Could not load elastic constants from {material_yaml_path}: {e}")
    
    return None, None, None


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


def detect_yield_point(
    strain: np.ndarray,
    stress_vm: np.ndarray,
    stress_el_cubic: np.ndarray,
    xi0_MPa: float,
    E_GPa: float,
    yield_threshold_ratio: float = 0.02
) -> Optional[int]:
    """
    Detect yield point using the actual yield stress (xi0) for each grain.
    Uses cubic elastic estimate for better accuracy.
    """
    if len(strain) < 3:
        return None
    
    # Expected yield strain based on material properties
    E_MPa = E_GPa * 1000
    expected_yield_strain = (xi0_MPa / E_MPa) * 100  # Convert to %
    
    # Method 1: Find where stress first reaches or exceeds yield stress (xi0)
    yield_stress_threshold = xi0_MPa * 0.95
    yield_by_stress = np.where(stress_vm >= yield_stress_threshold)[0]
    
    if len(yield_by_stress) > 0:
        return int(yield_by_stress[0])
    
    # Method 2: Find where stress deviates significantly from cubic elastic estimate
    deviation = np.abs((stress_vm - stress_el_cubic) / (stress_el_cubic + 1e-6))
    significant_deviation = np.where(deviation > 0.05)[0]  # 5% deviation
    
    if len(significant_deviation) > 0:
        return int(significant_deviation[0])
    
    # Method 3: Use expected yield strain
    if expected_yield_strain > 0 and expected_yield_strain < strain.max():
        closest_idx = np.argmin(np.abs(strain - expected_yield_strain))
        if stress_vm[closest_idx] >= xi0_MPa * 0.8:
            return int(closest_idx)
    
    return None


def classify_grain_plasticity(
    strain: np.ndarray,
    stress_vm: np.ndarray,
    stress_el_cubic: np.ndarray,
    threshold_ratio: float = 0.1
) -> Tuple[str, float]:
    """
    Classify grain as 'plastic' or 'elastic' based on deviation from cubic elastic estimate.
    """
    if len(strain) < 2:
        return 'elastic', 0.0
    
    # Deviation ratio: (actual - cubic elastic) / cubic elastic
    deviation_ratio = (stress_vm - stress_el_cubic) / (stress_el_cubic + 1e-6)
    mean_dev = np.abs(deviation_ratio).mean()
    
    if mean_dev > threshold_ratio:
        return 'plastic', mean_dev
    else:
        return 'elastic', mean_dev


def load_props(labels: np.ndarray, props_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load properties and map to per-grain arrays."""
    props_raw = np.load(props_path, allow_pickle=True)
    
    if props_raw.ndim == 0 and isinstance(props_raw.item(), dict):
        d = props_raw.item()
        E_arr = np.asarray(d['E'])
        nu_arr = np.asarray(d['nu'])
        xi0_arr = np.asarray(d['xi0'])
        h0_arr = np.asarray(d['h0'])
        props = np.stack([E_arr, nu_arr, xi0_arr, h0_arr], axis=1)
    else:
        props = props_raw
    
    max_label = labels.max()
    if max_label >= props.shape[0]:
        raise ValueError(f"labels contain grain id {max_label} but props only has {props.shape[0]} rows")
    
    E_map = props[labels, 0]
    nu_map = props[labels, 1]
    xi0_map = props[labels, 2]
    h0_map = props[labels, 3]
    
    return E_map, nu_map, xi0_map, h0_map


def main():
    ap = argparse.ArgumentParser(description='Enhanced per-grain stress-strain analysis with cubic elasticity')
    ap.add_argument('--seed', type=str, required=True, help='Seed ID (e.g., 105447566)')
    ap.add_argument('--hdf5-dir', type=str, default='simulation_results/hdf5_files', help='Directory with HDF5 files')
    ap.add_argument('--labels-dir', type=str, default='labels', help='Directory with label .npy files')
    ap.add_argument('--props-dir', type=str, default='props', help='Directory with props .npy files')
    ap.add_argument('--material-dir', type=str, default='material_yaml_fixed', help='Directory with material YAML files')
    ap.add_argument('--out', type=str, default='per_grain_analysis', help='Output directory')
    ap.add_argument('--threshold', type=float, default=0.1, help='Threshold ratio for plastic classification (default: 0.1)')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    seed = args.seed
    h5_path = os.path.join(args.hdf5_dir, f'seed{seed}.hdf5')
    material_yaml_path = os.path.join(args.material_dir, f'material_seed{seed}.yaml')
    
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
    if not os.path.isfile(material_yaml_path):
        print(f"Warning: Material YAML not found: {material_yaml_path}, using isotropic estimates only")
        material_yaml_path = None
    
    print(f"Loading data for seed {seed}...")
    labels = np.load(labels_path)
    if labels.ndim != 2:
        raise ValueError(f"Labels must be 2D, got shape {labels.shape}")
    
    E_map, nu_map, xi0_map, h0_map = load_props(labels, props_path)
    
    print(f"Extracting per-grain stress-strain data...")
    grain_data = load_per_grain_data(h5_path, labels)
    
    unique_grains = sorted(grain_data.keys())
    print(f"Found {len(unique_grains)} grains")
    
    # Load elastic constants from material YAML if available
    elastic_constants = {}
    if material_yaml_path and os.path.isfile(material_yaml_path):
        print(f"Loading elastic constants from {material_yaml_path}...")
        for gid in unique_grains:
            C11, C12, C44 = load_elastic_constants(material_yaml_path, gid)
            if C11 is not None:
                elastic_constants[gid] = (C11, C12, C44)
    
    # Classify grains
    grain_classifications = {}
    for gid in unique_grains:
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        
        if len(strain) == 0:
            continue
        
        # Get grain properties
        mask = (labels == gid)
        E_grain = E_map[mask].mean()
        nu_grain = nu_map[mask].mean()
        xi0_grain = xi0_map[mask].mean()
        h0_grain = h0_map[mask].mean()
        
        # Compute elastic estimates
        strain_frac = strain / 100.0
        stress_el_iso = compute_isotropic_elastic_stress(E_grain, nu_grain, strain_frac)
        
        # Use cubic elastic if available, otherwise isotropic
        if gid in elastic_constants:
            C11, C12, C44 = elastic_constants[gid]
            stress_el_cubic = compute_cubic_elastic_stress(C11, C12, C44, strain_frac)
            stress_el = stress_el_cubic  # Use cubic for classification
        else:
            stress_el_cubic = None
            stress_el = stress_el_iso  # Fallback to isotropic
        
        classification, dev_ratio = classify_grain_plasticity(
            strain, stress_vm, stress_el, threshold_ratio=args.threshold
        )
        
        grain_classifications[gid] = {
            'classification': classification,
            'deviation_ratio': float(dev_ratio),
            'E': float(E_grain),
            'nu': float(nu_grain),
            'xi0': float(xi0_grain),
            'h0': float(h0_grain),
            'C11': float(elastic_constants[gid][0]) if gid in elastic_constants else None,
            'C12': float(elastic_constants[gid][1]) if gid in elastic_constants else None,
            'C44': float(elastic_constants[gid][2]) if gid in elastic_constants else None,
        }
    
    # Separate plastic and elastic grains
    plastic_grains = [gid for gid, info in grain_classifications.items() if info['classification'] == 'plastic']
    elastic_grains = [gid for gid, info in grain_classifications.items() if info['classification'] == 'elastic']
    
    print(f"\nClassification Results:")
    print(f"  Plastic grains: {len(plastic_grains)} ({len(plastic_grains)/len(unique_grains)*100:.1f}%)")
    print(f"  Elastic grains: {len(elastic_grains)} ({len(elastic_grains)/len(unique_grains)*100:.1f}%)")
    
    # Create enhanced visualization with both isotropic and cubic elastic estimates
    all_grains_sorted = (
        sorted(plastic_grains, key=lambda g: grain_classifications[g]['deviation_ratio'], reverse=True) +
        sorted(elastic_grains, key=lambda g: grain_classifications[g]['deviation_ratio'])
    )
    
    total_grains = len(all_grains_sorted)
    if total_grains == 0:
        print("No grains to plot!")
        return
    
    num_cols = int(np.ceil(np.sqrt(total_grains * 1.5)))
    num_rows = int(np.ceil(total_grains / num_cols))
    
    fig2, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows))
    if total_grains == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot all grains with enhanced elastic estimates
    for idx, gid in enumerate(all_grains_sorted):
        ax = axes[idx]
        strain = np.array(grain_data[gid]['strain'])
        stress_vm = np.array(grain_data[gid]['stress_vm'])
        E_g = grain_classifications[gid]['E']
        nu_g = grain_classifications[gid]['nu']
        classification = grain_classifications[gid]['classification']
        dev_ratio = grain_classifications[gid]['deviation_ratio']
        xi0_grain = grain_classifications[gid]['xi0'] / 1e6  # Convert to MPa
        E_grain_GPa = grain_classifications[gid]['E'] / 1e9  # Convert to GPa
        
        strain_frac = strain / 100.0
        
        # Compute both elastic estimates
        stress_el_iso = compute_isotropic_elastic_stress(E_g, nu_g, strain_frac)
        
        if gid in elastic_constants:
            C11, C12, C44 = elastic_constants[gid]
            stress_el_cubic = compute_cubic_elastic_stress(C11, C12, C44, strain_frac)
            has_cubic = True
        else:
            stress_el_cubic = None
            has_cubic = False
        
        # Detect yield point using cubic elastic if available, otherwise isotropic
        stress_el_for_yield = stress_el_cubic if has_cubic else stress_el_iso
        yield_idx = detect_yield_point(strain, stress_vm, stress_el_for_yield, xi0_grain, E_grain_GPa, yield_threshold_ratio=0.02)
        
        # Plot elastic estimates
        ax.plot(strain, stress_el_iso, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, 
               label='Isotropic (E×ε)', zorder=1)
        
        if has_cubic:
            ax.plot(strain, stress_el_cubic, color='purple', linestyle=':', linewidth=2, alpha=0.7, 
                   label='Cubic (C11/C12/C44)', zorder=2)
        
        # Plot actual data
        color = 'red' if classification == 'plastic' else 'blue'
        
        if classification == 'plastic' and yield_idx is not None and yield_idx > 0:
            # Plot pre-yield region
            if yield_idx < len(strain):
                ax.plot(strain[:yield_idx+1], stress_vm[:yield_idx+1], 
                       color='green', linewidth=3, alpha=0.9, 
                       label='Pre-Yield (GT)', zorder=3)
            # Plot hardening region
            if yield_idx < len(strain) - 1:
                ax.plot(strain[yield_idx:], stress_vm[yield_idx:], 
                       color='red', linewidth=3, alpha=0.9, label='Hardening (GT)', zorder=4)
        else:
            ax.plot(strain, stress_vm, color=color, linewidth=2.5, label='Actual (GT)', zorder=3)
        
        # Mark yield point
        if yield_idx is not None and yield_idx < len(strain):
            ax.plot(strain[yield_idx], stress_vm[yield_idx], 'ko', markersize=10, 
                   label='Yield Point', zorder=5, markerfacecolor='yellow', markeredgewidth=2)
            ax.axvline(strain[yield_idx], color='orange', linestyle=':', linewidth=2, alpha=0.8, zorder=2)
        
        title = f'Grain {gid} ({classification.capitalize()}, dev={dev_ratio:.3f})'
        if yield_idx is not None:
            title += f'\nYield at {strain[yield_idx]:.3f}%'
        if has_cubic:
            title += f'\nCubic: C11={grain_classifications[gid]["C11"]/1e9:.0f}, C12={grain_classifications[gid]["C12"]/1e9:.0f}, C44={grain_classifications[gid]["C44"]/1e9:.0f} GPa'
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
    out_file2 = os.path.join(args.out, f'seed{seed}_individual_grains_enhanced.png')
    plt.savefig(out_file2, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Enhanced individual grains: {out_file2}")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Enhanced per-grain analysis complete for seed {seed}!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

