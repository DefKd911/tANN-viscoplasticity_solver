"""
Plot Stress-Strain curve from DAMASK HDF5 output (using F and P)
Computes Cauchy stress from F and P, then plots average stress vs strain
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional, Tuple

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


def compute_cauchy_stress(F, P):
    """
    Compute Cauchy stress from F and P.
    sigma = (1/det(F)) * P * F^T
    """
    if F.ndim == 3:  # (N, 3, 3)
        N = F.shape[0]
        sigma = np.zeros_like(P)
        
        for i in range(N):
            F_i = F[i]
            P_i = P[i]
            det_F = np.linalg.det(F_i)
            if abs(det_F) > 1e-10:
                sigma[i] = (1.0 / det_F) * P_i @ F_i.T
        return sigma
    else:
        raise ValueError(f"Unexpected F shape: {F.shape}")

def von_mises_stress(sigma):
    """
    Compute von Mises stress from Cauchy stress tensor.
    sigma_vM = sqrt(3/2 * s_ij * s_ij) where s_ij is deviatoric stress
    """
    if sigma.ndim == 3:  # (N, 3, 3)
        N = sigma.shape[0]
        sigma_vM = np.zeros(N)
        
        for i in range(N):
            s = sigma[i]
            # Deviatoric stress
            s_dev = s - np.trace(s) / 3.0 * np.eye(3)
            # Von Mises
            sigma_vM[i] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
        return sigma_vM
    else:
        raise ValueError(f"Unexpected sigma shape: {sigma.shape}")

def extract_strain_from_F(F):
    """
    Extract engineering strain from deformation gradient F.
    epsilon_eng = F11 - 1 (for uniaxial tension in x-direction)
    """
    if F.ndim == 3:  # (N, 3, 3)
        # Average F11 across all spatial points
        F11 = F[:, 0, 0].mean()
        return F11 - 1.0
    else:
        raise ValueError(f"Unexpected F shape: {F.shape}")


def _reshape_tensor(arr: np.ndarray) -> np.ndarray:
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


def _load_tensor(f: h5py.File, base: str, rel_paths) -> Optional[np.ndarray]:
    for rel in rel_paths:
        path = f"{base}/{rel}"
        data = _load_dataset(f, path)
        if data is not None:
            return _reshape_tensor(data)
    return None


def load_sigma_tensor(f: h5py.File, base: str) -> Optional[np.ndarray]:
    return _load_tensor(f, base, SIGMA_DATASETS)


def load_F_P_tensors(f: h5py.File, base: str) -> Tuple[np.ndarray, np.ndarray]:
    F = _load_tensor(f, base, F_DATASETS)
    P = _load_tensor(f, base, P_DATASETS)
    if F is None or P is None:
        raise KeyError("F/P tensors missing")
    return F, P

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_ss_curve_from_F_P.py <hdf5_file>")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    
    print(f"\n{'='*70}")
    print(f"Extracting Stress-Strain Curve from DAMASK HDF5")
    print(f"{'='*70}")
    print(f"File: {h5_file}")
    print(f"{'='*70}\n")
    
    strains = []
    stress_avg = []
    stress_vM_avg = []
    
    with h5py.File(h5_file, 'r') as f:
        # Get all increments
        increments = sorted([k for k in f.keys() if k.startswith('increment_')],
                          key=lambda x: int(x.split('_')[1]))
        
        print(f"Found {len(increments)} increments: {increments[0]} to {increments[-1]}")
        print(f"\nProcessing increments...")
        
        for inc_name in increments:
            inc_num = int(inc_name.split('_')[1])
            
            # Path to homogenization data
            hom_path = f'{inc_name}/homogenization/h0'
            
            if f"{hom_path}/mechanical" not in f:
                print(f"  [SKIP] {inc_name}: No homogenization data")
                continue
            
            sigma = load_sigma_tensor(f, hom_path)
            if sigma is None:
                F, P = load_F_P_tensors(f, hom_path)
                strain = extract_strain_from_F(F)
                sigma = compute_cauchy_stress(F, P)
            else:
                # Need strain for plotting; derive from F if possible
                F = _load_tensor(f, hom_path, F_DATASETS)
                strain = extract_strain_from_F(F) if F is not None else 0.0
            
            # Average stress (11 component for uniaxial tension)
            stress_11_avg = sigma[:, 0, 0].mean() / 1e6  # Convert to MPa
            
            # Von Mises stress average
            sigma_vM = von_mises_stress(sigma)
            sigma_vM_avg_val = sigma_vM.mean() / 1e6  # Convert to MPa
            
            strains.append(strain * 100)  # Convert to percent
            stress_avg.append(stress_11_avg)
            stress_vM_avg.append(sigma_vM_avg_val)
            
            print(f"  Inc {inc_num:2d}: Strain={strain*100:.3f}%, "
                  f"Stress_11={stress_11_avg:.1f} MPa, "
                  f"Stress_vM={sigma_vM_avg_val:.1f} MPa")
    
    # Convert to arrays
    strains = np.array(strains)
    stress_avg = np.array(stress_avg)
    stress_vM_avg = np.array(stress_vM_avg)
    
    print(f"\n{'='*70}")
    print(f"Stress-Strain Summary")
    print(f"{'='*70}")
    print(f"Strain range:     {strains.min():.3f}% to {strains.max():.3f}%")
    print(f"Stress_11 range:  {stress_avg.min():.1f} to {stress_avg.max():.1f} MPa")
    print(f"Stress_vM range:  {stress_vM_avg.min():.1f} to {stress_vM_avg.max():.1f} MPa")
    print(f"{'='*70}\n")
    
    # Plot Stress-Strain curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Stress_11 vs Strain
    ax1.plot(strains, stress_avg, 'b-o', linewidth=2, markersize=6, label='Sigma_11')
    ax1.set_xlabel('Engineering Strain (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stress_11 (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title('Uniaxial Stress vs Strain', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: von Mises stress vs Strain
    ax2.plot(strains, stress_vM_avg, 'r-s', linewidth=2, markersize=6, label='Von Mises Stress')
    ax2.set_xlabel('Engineering Strain (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Von Mises Stress (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title('Von Mises Stress vs Strain', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Create stress_curves directory if it doesn't exist
    os.makedirs('simulation_results/stress_curves', exist_ok=True)
    
    # Extract seed name from HDF5 file path
    seed_name = os.path.basename(h5_file).replace('.hdf5', '')
    
    # Save figure to stress_curves directory
    output_file = f'simulation_results/stress_curves/{seed_name}_stress_strain.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Stress-strain curve: {output_file}")
    
    # Also save data to CSV in stress_curves directory
    csv_file = f'simulation_results/stress_curves/{seed_name}_stress_strain.csv'
    with open(csv_file, 'w') as f:
        f.write("Increment,Strain(%),Stress_11(MPa),Stress_vM(MPa)\n")
        for i, (strain, s11, svm) in enumerate(zip(strains, stress_avg, stress_vM_avg)):
            f.write(f"{i},{strain:.6f},{s11:.6f},{svm:.6f}\n")
    print(f"[SAVED] CSV data: {csv_file}")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Stress-strain analysis complete!")
    print(f"{'='*70}\n")

