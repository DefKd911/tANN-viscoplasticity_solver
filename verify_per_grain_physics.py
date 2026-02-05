#!/usr/bin/env python3
"""
Verify per-grain stress-strain curve physics and explain the visualization.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json

def load_grain_data(h5_path, labels_path, props_path, grain_id):
    """Load stress-strain data for a specific grain."""
    labels = np.load(labels_path)
    props_raw = np.load(props_path, allow_pickle=True)
    
    if props_raw.ndim == 0 and isinstance(props_raw.item(), dict):
        d = props_raw.item()
        props = np.stack([np.asarray(d['E']), np.asarray(d['nu']), 
                         np.asarray(d['xi0']), np.asarray(d['h0'])], axis=1)
    else:
        props = props_raw
    
    E_g = props[grain_id, 0]
    nu_g = props[grain_id, 1]
    xi0_g = props[grain_id, 2] / 1e6  # Convert to MPa
    h0_g = props[grain_id, 2] / 1e9  # Convert to GPa
    
    # Load HDF5 data
    strains = []
    stress_vm = []
    stress_11 = []
    
    H, W = labels.shape
    N_voxels = H * W
    mask = (labels == grain_id)
    
    with h5py.File(h5_path, 'r') as f:
        increments = sorted([k for k in f.keys() if k.startswith('increment_')],
                          key=lambda x: int(x.split('_')[1]))
        
        for inc_name in increments:
            hom_path = f'{inc_name}/homogenization/h0'
            if f"{hom_path}/mechanical" not in f:
                continue
            
            # Use the same logic as analyze_per_grain_ss_curves.py
            # Load stress tensor
            sigma = None
            sigma_paths = [
                f"{hom_path}/mechanical/output/stress_Cauchy",
                f"{hom_path}/mechanical/stress_Cauchy",
            ]
            for path in sigma_paths:
                if path in f:
                    data = f[path][()]
                    # Reshape to (N, 3, 3) where N = H*W
                    if data.ndim == 5:  # (H, W, 1, 3, 3)
                        data = data[:, :, 0, :, :]
                    if data.ndim == 4:  # (H, W, 3, 3)
                        sigma = data.reshape(-1, 3, 3)
                    break
            
            # Load F and P if sigma not found
            F_flat = None
            P_flat = None
            if sigma is None:
                F_paths = [f"{hom_path}/mechanical/output/F", f"{hom_path}/mechanical/F"]
                P_paths = [f"{hom_path}/mechanical/output/P", f"{hom_path}/mechanical/P"]
                for path in F_paths:
                    if path in f:
                        F_data = f[path][()]
                        if F_data.ndim == 5:
                            F_data = F_data[:, :, 0, :, :]
                        if F_data.ndim == 4:
                            F_flat = F_data.reshape(-1, 3, 3)
                        break
                for path in P_paths:
                    if path in f:
                        P_data = f[path][()]
                        if P_data.ndim == 5:
                            P_data = P_data[:, :, 0, :, :]
                        if P_data.ndim == 4:
                            P_flat = P_data.reshape(-1, 3, 3)
                        break
                
                if F_flat is not None and P_flat is not None:
                    # Compute Cauchy stress
                    N = F_flat.shape[0]
                    sigma = np.zeros((N, 3, 3))
                    for i in range(N):
                        det_F = np.linalg.det(F_flat[i])
                        if abs(det_F) > 1e-10:
                            sigma[i] = (1.0 / det_F) * P_flat[i] @ F_flat[i].T
            
            if sigma is None:
                continue
            
            # Get F for strain (if not already loaded)
            if F_flat is None:
                for path in [f"{hom_path}/mechanical/output/F", f"{hom_path}/mechanical/F"]:
                    if path in f:
                        F_data = f[path][()]
                        if F_data.ndim == 5:
                            F_data = F_data[:, :, 0, :, :]
                        if F_data.ndim == 4:
                            F_flat = F_data.reshape(-1, 3, 3)
                        break
            
            if F_flat is None:
                continue
            
            # Compute per-voxel quantities
            strain_11 = F_flat[:, 0, 0] - 1.0  # Engineering strain (N,)
            
            # Von Mises stress
            vm = np.zeros(sigma.shape[0])
            for i in range(sigma.shape[0]):
                s = sigma[i]
                s_dev = s - np.trace(s) / 3.0 * np.eye(3)
                vm[i] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
            
            # Reshape back to spatial grid for masking
            strain_11_2d = strain_11.reshape(H, W)
            vm_2d = vm.reshape(H, W)
            sigma_11_2d = sigma[:, 0, 0].reshape(H, W)
            
            # Average over grain
            strain_avg = strain_11_2d[mask].mean() * 100  # Convert to %
            vm_avg = vm_2d[mask].mean() / 1e6  # Convert to MPa
            s11_avg = sigma_11_2d[mask].mean() / 1e6  # Convert to MPa
            
            strains.append(strain_avg)
            stress_vm.append(vm_avg)
            stress_11.append(s11_avg)
    
    return np.array(strains), np.array(stress_vm), np.array(stress_11), E_g, nu_g, xi0_g, h0_g


def main():
    seed = "105447566"
    h5_path = f"simulation_results/hdf5_files/seed{seed}.hdf5"
    labels_path = f"labels/labels_seed{seed}.npy"
    props_path = f"props/props_seed{seed}.npy"
    summary_path = f"per_grain_analysis/seed{seed}_summary.json"
    
    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Analyze a few representative grains
    grains_to_check = [3, 0, 1]  # Highly plastic, elastic, moderate plastic
    
    fig, axes = plt.subplots(len(grains_to_check), 2, figsize=(14, 4*len(grains_to_check)))
    if len(grains_to_check) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, gid in enumerate(grains_to_check):
        gid_str = str(gid)
        if gid_str not in summary['grain_classifications']:
            continue
        
        info = summary['grain_classifications'][gid_str]
        strains, stress_vm, stress_11, E, nu, xi0, h0 = load_grain_data(
            h5_path, labels_path, props_path, gid
        )
        
        # Convert units
        E_GPa = E / 1e9
        xi0_MPa = xi0
        h0_GPa = h0
        
        # Elastic estimate
        strain_frac = strains / 100.0
        stress_el = (E * strain_frac) / 1e6  # MPa
        
        # Theoretical yield stress (for J2 plasticity)
        sigma_y = xi0_MPa  # Initial flow stress
        
        # Plot 1: Full curve with analysis
        ax1 = axes[idx, 0]
        ax1.plot(strains, stress_vm, 'b-', linewidth=2.5, label='Actual σvM', zorder=3)
        ax1.plot(strains, stress_el, 'r--', linewidth=2, alpha=0.7, label=f'Elastic (E={E_GPa:.0f} GPa)', zorder=1)
        ax1.axhline(sigma_y, color='orange', linestyle=':', linewidth=2, 
                   label=f'Yield (xi0={xi0_MPa:.0f} MPa)', zorder=2)
        
        # Mark yield point
        yield_idx = np.where(stress_vm >= sigma_y)[0]
        if len(yield_idx) > 0:
            first_yield = yield_idx[0]
            ax1.plot(strains[first_yield], stress_vm[first_yield], 'ko', 
                    markersize=10, markerfacecolor='yellow', markeredgewidth=2, zorder=4)
            ax1.axvline(strains[first_yield], color='gray', linestyle=':', alpha=0.5)
        
        ax1.set_xlabel('Strain (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Stress (MPa)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Grain {gid} - {info["classification"].upper()}\n'
                     f'E={E_GPa:.0f} GPa, ν={nu:.3f}, xi0={xi0_MPa:.0f} MPa, h0={h0_GPa:.1f} GPa\n'
                     f'Deviation={info["deviation_ratio"]:.3f}', 
                     fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='best')
        
        # Plot 2: Deviation analysis
        ax2 = axes[idx, 1]
        deviation = (stress_vm - stress_el) / (stress_el + 1e-6) * 100  # Percentage
        ax2.plot(strains, deviation, 'g-', linewidth=2, label='% Deviation from Elastic')
        ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(10, color='orange', linestyle=':', linewidth=1.5, label='10% threshold')
        ax2.axhline(-10, color='orange', linestyle=':', linewidth=1.5)
        
        ax2.set_xlabel('Strain (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Deviation (%)', fontsize=11, fontweight='bold')
        mean_dev = np.abs(deviation).mean() if len(deviation) > 0 else 0.0
        ax2.set_title(f'Grain {gid} - Deviation Analysis\n'
                     f'Mean |dev| = {mean_dev:.1f}%', 
                     fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Print diagnostics
        print(f"\n{'='*70}")
        print(f"Grain {gid} Analysis:")
        print(f"{'='*70}")
        print(f"Properties: E={E_GPa:.0f} GPa, nu={nu:.3f}, xi0={xi0_MPa:.0f} MPa, h0={h0_GPa:.1f} GPa")
        print(f"Classification: {info['classification']} (dev={info['deviation_ratio']:.3f})")
        print(f"Strain range: {strains.min():.3f}% to {strains.max():.3f}%")
        print(f"Stress range: {stress_vm.min():.1f} to {stress_vm.max():.1f} MPa")
        print(f"Elastic estimate at max strain: {stress_el[-1]:.1f} MPa")
        print(f"Actual stress at max strain: {stress_vm[-1]:.1f} MPa")
        print(f"Yield stress (xi0): {sigma_y:.1f} MPa")
        if len(yield_idx) > 0:
            print(f"First yield detected at: {strains[first_yield]:.3f}% strain, {stress_vm[first_yield]:.1f} MPa")
        mean_dev = np.abs(deviation).mean() if len(deviation) > 0 else 0.0
        print(f"Mean deviation: {mean_dev:.1f}%")
    
    plt.tight_layout()
    os.makedirs('per_grain_analysis', exist_ok=True)
    plt.savefig('per_grain_analysis/physics_verification.png', dpi=300, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"[SAVED] Physics verification: per_grain_analysis/physics_verification.png")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

