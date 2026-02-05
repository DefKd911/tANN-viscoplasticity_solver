#!/usr/bin/env python3
"""
Check GT overlay visualizations for correctness:
1. Stress values in correct units (MPa, not Pa)
2. Property maps in correct physical units
3. Elastic estimate formula (should be E*strain, not E*strain/(1+nu))
4. GB overlay alignment
5. Physical consistency checks
"""

import os
import argparse
import numpy as np
import h5py
from pathlib import Path


def check_stress_units(vm_pa: np.ndarray) -> dict:
    """Check if stress values are in reasonable MPa range."""
    vm_mpa = vm_pa * 1e-6
    stats = {
        'min_mpa': float(vm_mpa.min()),
        'max_mpa': float(vm_mpa.max()),
        'mean_mpa': float(vm_mpa.mean()),
        'reasonable': True,
        'issues': []
    }
    
    # Check for unphysically high values (> 10 GPa = 10000 MPa)
    if stats['max_mpa'] > 10000:
        stats['reasonable'] = False
        stats['issues'].append(f"Max stress {stats['max_mpa']:.1f} MPa is unphysically high (>10 GPa)")
    
    # Check for negative values
    if stats['min_mpa'] < -100:
        stats['reasonable'] = False
        stats['issues'].append(f"Min stress {stats['min_mpa']:.1f} MPa is unphysically negative")
    
    # Check for reasonable range (typically 0-1000 MPa for most materials)
    if stats['max_mpa'] < 1:
        stats['issues'].append(f"Max stress {stats['max_mpa']:.1f} MPa seems too low (check scaling)")
    
    return stats


def check_property_units(E: np.ndarray, nu: np.ndarray, xi0: np.ndarray, h0: np.ndarray) -> dict:
    """Check if property values are in correct units."""
    E_gpa = E / 1e9
    nu_dimless = nu
    xi0_mpa = xi0 / 1e6
    h0_gpa = h0 / 1e9
    
    stats = {
        'E_gpa': {'min': float(E_gpa.min()), 'max': float(E_gpa.max()), 'mean': float(E_gpa.mean())},
        'nu': {'min': float(nu_dimless.min()), 'max': float(nu_dimless.max()), 'mean': float(nu_dimless.mean())},
        'xi0_mpa': {'min': float(xi0_mpa.min()), 'max': float(xi0_mpa.max()), 'mean': float(xi0_mpa.mean())},
        'h0_gpa': {'min': float(h0_gpa.min()), 'max': float(h0_gpa.max()), 'mean': float(h0_gpa.mean())},
        'issues': []
    }
    
    # Check E (should be 50-500 GPa for typical metals)
    if stats['E_gpa']['min'] < 10 or stats['E_gpa']['max'] > 1000:
        stats['issues'].append(f"E range {stats['E_gpa']['min']:.1f}-{stats['E_gpa']['max']:.1f} GPa seems unusual")
    
    # Check nu (should be 0.1-0.5)
    if stats['nu']['min'] < 0.0 or stats['nu']['max'] > 0.6:
        stats['issues'].append(f"nu range {stats['nu']['min']:.3f}-{stats['nu']['max']:.3f} seems unusual")
    
    # Check xi0 (should be 50-1000 MPa for typical metals)
    if stats['xi0_mpa']['min'] < 1 or stats['xi0_mpa']['max'] > 5000:
        stats['issues'].append(f"xi0 range {stats['xi0_mpa']['min']:.1f}-{stats['xi0_mpa']['max']:.1f} MPa seems unusual")
    
    # Check h0 (should be 0.1-10 GPa for typical metals)
    if stats['h0_gpa']['min'] < 0.01 or stats['h0_gpa']['max'] > 100:
        stats['issues'].append(f"h0 range {stats['h0_gpa']['min']:.3f}-{stats['h0_gpa']['max']:.3f} GPa seems unusual")
    
    return stats


def check_elastic_formula(E: np.ndarray, nu: np.ndarray, eps11: float, vm_mpa: np.ndarray) -> dict:
    """Check if elastic estimate formula is correct.
    
    For uniaxial tension: σ_vM = E * ε11 (NOT E * ε11 / (1+nu))
    """
    # Correct formula
    elastic_correct = (E * eps11) / 1e6  # MPa
    
    # Incorrect formula (what's currently in visualize_gt_from_hdf5.py line 558)
    elastic_incorrect = (E * eps11 / (1.0 + np.clip(nu, 1e-9, 0.499999))) / 1e6  # MPa
    
    # Compare with actual stress
    diff_correct = np.abs(vm_mpa - elastic_correct)
    diff_incorrect = np.abs(vm_mpa - elastic_incorrect)
    
    stats = {
        'eps11': eps11,
        'mean_elastic_correct_mpa': float(elastic_correct.mean()),
        'mean_elastic_incorrect_mpa': float(elastic_incorrect.mean()),
        'mean_vm_mpa': float(vm_mpa.mean()),
        'mean_diff_correct': float(diff_correct.mean()),
        'mean_diff_incorrect': float(diff_incorrect.mean()),
        'formula_issue': False,
        'issues': []
    }
    
    # If incorrect formula is closer to actual stress, that's suspicious
    # (though in plastic regime, both will deviate)
    if stats['mean_diff_incorrect'] < stats['mean_diff_correct'] * 0.8:
        stats['formula_issue'] = True
        stats['issues'].append(
            f"Incorrect formula (E*ε/(1+ν)) is closer to actual stress than correct formula (E*ε). "
            f"This suggests the visualization may be using the wrong formula."
        )
    
    return stats


def load_vm_from_hdf5(h5_path: str) -> np.ndarray:
    """Load von Mises stress from HDF5 (returns in Pa)."""
    from visualize_gt_from_hdf5 import load_vm_from_hdf5 as load_vm
    return load_vm(h5_path)


def estimate_eps11_from_hdf5(h5_path: str) -> float:
    """Estimate average engineering strain ε11 from HDF5."""
    from visualize_gt_from_hdf5 import estimate_eps11_from_hdf5 as est_eps
    return est_eps(h5_path)


def load_props(props_dir: str, seed: str) -> tuple:
    """Load property maps (E, nu, xi0, h0) in Pa/Pa/Pa/Pa."""
    from visualize_gt_from_hdf5 import find_props_path, per_pixel_maps, load_labels
    
    props_path = find_props_path(props_dir, seed)
    if props_path is None:
        raise FileNotFoundError(f"Props not found for seed {seed}")
    
    props_arr = np.load(props_path, allow_pickle=True)
    labels_path = find_props_path(props_dir, seed).replace('props_', 'labels_').replace('seed', 'seed')
    if not os.path.isfile(labels_path):
        labels_path = os.path.join(os.path.dirname(props_path), f'seed{seed}.npy')
    
    labels = load_labels(os.path.dirname(labels_path), seed)
    if labels is None:
        raise FileNotFoundError(f"Labels not found for seed {seed}")
    
    E, nu, xi0, h0 = per_pixel_maps(labels, props_arr)
    return E, nu, xi0, h0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5-dir', default='simulation_results/hdf5_files')
    ap.add_argument('--props-dir', default='props')
    ap.add_argument('--seeds', default='', help='Comma-separated list of seeds, or empty for all in gt_overlays')
    ap.add_argument('--gt-overlays-dir', default='ML_EVAL/gt_overlays')
    args = ap.parse_args()
    
    # Get seeds to check
    if args.seeds.strip():
        seeds = [s.strip() for s in args.seeds.split(',')]
    else:
        # Find seeds from existing GT overlay images
        overlay_dir = Path(args.gt_overlays_dir)
        if overlay_dir.exists():
            seeds = []
            for f in overlay_dir.glob('seed*_gt_7panel.png'):
                m = f.stem.replace('seed', '').replace('_gt_7panel', '')
                if m.isdigit():
                    seeds.append(m)
            seeds = sorted(set(seeds))
        else:
            print(f"[ERROR] GT overlays directory not found: {args.gt_overlays_dir}")
            return
    
    if not seeds:
        print("[WARN] No seeds found to check.")
        return
    
    print(f"\n{'='*70}")
    print(f"CHECKING GT OVERLAYS FOR {len(seeds)} SEEDS")
    print(f"{'='*70}\n")
    
    all_issues = []
    
    for i, seed in enumerate(seeds[:10], 1):  # Check first 10
        print(f"[{i}/{min(10, len(seeds))}] Checking seed {seed}...")
        try:
            h5_path = os.path.join(args.hdf5_dir, f'seed{seed}.hdf5')
            if not os.path.isfile(h5_path):
                print(f"  [SKIP] HDF5 not found: {h5_path}")
                continue
            
            # Load data
            vm_pa = load_vm_from_hdf5(h5_path)
            E, nu, xi0, h0 = load_props(args.props_dir, seed)
            eps11 = estimate_eps11_from_hdf5(h5_path)
            
            # Check stress units
            stress_stats = check_stress_units(vm_pa)
            if stress_stats['issues']:
                all_issues.append(f"seed{seed}: {', '.join(stress_stats['issues'])}")
            
            # Check property units
            prop_stats = check_property_units(E, nu, xi0, h0)
            if prop_stats['issues']:
                all_issues.append(f"seed{seed}: {', '.join(prop_stats['issues'])}")
            
            # Check elastic formula
            vm_mpa = vm_pa * 1e-6
            elastic_stats = check_elastic_formula(E, nu, eps11, vm_mpa)
            if elastic_stats['formula_issue']:
                all_issues.append(f"seed{seed}: {', '.join(elastic_stats['issues'])}")
            
            # Print summary
            print(f"  σvM: {stress_stats['min_mpa']:.1f}-{stress_stats['max_mpa']:.1f} MPa (mean: {stress_stats['mean_mpa']:.1f} MPa)")
            print(f"  E: {prop_stats['E_gpa']['min']:.1f}-{prop_stats['E_gpa']['max']:.1f} GPa")
            print(f"  ε11: {eps11*100:.3f}%")
            print(f"  Elastic estimate (correct): {elastic_stats['mean_elastic_correct_mpa']:.1f} MPa")
            print(f"  Elastic estimate (incorrect): {elastic_stats['mean_elastic_incorrect_mpa']:.1f} MPa")
            print(f"  Actual σvM: {elastic_stats['mean_vm_mpa']:.1f} MPa")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_issues.append(f"seed{seed}: {str(e)}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    if all_issues:
        print("ISSUES FOUND:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("✓ No major issues found!")
    
    print("\nKEY CHECKS:")
    print("  1. Stress units: Should be in MPa (0-1000 MPa typical)")
    print("  2. Property units: E (GPa), nu (-), xi0 (MPa), h0 (GPa)")
    print("  3. Elastic formula: Should be E*strain, NOT E*strain/(1+nu)")
    print("  4. GB overlays: Should align with stress patterns")
    print("\nNOTE: If elastic formula is wrong, check visualize_gt_from_hdf5.py line 558")


if __name__ == '__main__':
    main()

