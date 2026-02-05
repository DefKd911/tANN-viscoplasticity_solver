"""
Extract Cauchy stress from F (deformation gradient) and P (1st Piola-Kirchhoff stress)
σ_Cauchy = (1/det(F)) * P * F^T
"""
import h5py
import numpy as np
import sys

def compute_cauchy_stress(F, P):
    """
    Compute Cauchy stress from deformation gradient F and 1st Piola-Kirchhoff stress P.
    σ = (1/det(F)) * P * F^T
    
    Args:
        F: Deformation gradient tensor (3x3 or array of 3x3)
        P: 1st Piola-Kirchhoff stress (3x3 or array of 3x3)
    
    Returns:
        Cauchy stress tensor
    """
    # Flatten spatial dimensions
    if F.ndim == 3:  # (N, 3, 3) format
        N = F.shape[0]
        sigma = np.zeros_like(P)
        
        for i in range(N):
            F_i = F[i]
            P_i = P[i]
            det_F = np.linalg.det(F_i)
            sigma[i] = (1.0 / det_F) * P_i @ F_i.T
        return sigma
    elif F.ndim == 5:  # (x, y, z, 3, 3)
        nx, ny, nz = F.shape[:3]
        sigma = np.zeros_like(P)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    F_ijk = F[i, j, k]
                    P_ijk = P[i, j, k]
                    det_F = np.linalg.det(F_ijk)
                    sigma[i, j, k] = (1.0 / det_F) * P_ijk @ F_ijk.T
        return sigma
    else:
        # Single tensor
        det_F = np.linalg.det(F)
        return (1.0 / det_F) * P @ F.T

def von_mises_stress(sigma):
    """
    Compute von Mises stress from Cauchy stress tensor.
    σ_vM = sqrt(3/2 * s_ij * s_ij)
    where s_ij is deviatoric stress
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
    elif sigma.ndim == 5:
        nx, ny, nz = sigma.shape[:3]
        sigma_vM = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    s = sigma[i, j, k]
                    # Deviatoric stress
                    s_dev = s - np.trace(s) / 3.0 * np.eye(3)
                    # Von Mises
                    sigma_vM[i, j, k] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
        return sigma_vM
    else:
        s_dev = sigma - np.trace(sigma) / 3.0 * np.eye(3)
        return np.sqrt(1.5 * np.sum(s_dev * s_dev))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_stress_from_F_P.py <hdf5_file> [increment]")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    increment = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print(f"\n{'='*60}")
    print(f"Extracting Cauchy stress from F and P")
    print(f"File: {h5_file}")
    print(f"Increment: {increment}")
    print(f"{'='*60}\n")
    
    with h5py.File(h5_file, 'r') as f:
        inc_name = f'increment_{increment}'
        
        if inc_name not in f:
            print(f"[ERROR] {inc_name} not found!")
            print(f"Available: {list(f.keys())}")
            sys.exit(1)
        
        # Try homogenization level (macro stress)
        hom_path = f'{inc_name}/homogenization/h0/mechanical'
        
        if hom_path not in f:
            print(f"[ERROR] Path not found: {hom_path}")
            sys.exit(1)
        
        F = f[f'{hom_path}/F'][()]
        P = f[f'{hom_path}/P'][()]
        
        print(f"F shape: {F.shape}")
        print(f"P shape: {P.shape}")
        
        # Compute Cauchy stress
        sigma = compute_cauchy_stress(F, P)
        
        print(f"\n{'='*60}")
        print(f"Cauchy Stress Statistics (MPa)")
        print(f"{'='*60}")
        print(f"Min:  {sigma.min() / 1e6:.2f} MPa")
        print(f"Max:  {sigma.max() / 1e6:.2f} MPa")
        print(f"Mean: {sigma.mean() / 1e6:.2f} MPa")
        
        # Compute von Mises
        sigma_vM = von_mises_stress(sigma)
        
        print(f"\n{'='*60}")
        print(f"Von Mises Stress (sigma_vM)")
        print(f"{'='*60}")
        print(f"Min:  {sigma_vM.min() / 1e6:.2f} MPa")
        print(f"Max:  {sigma_vM.max() / 1e6:.2f} MPa")
        print(f"Mean: {sigma_vM.mean() / 1e6:.2f} MPa")
        
        # Sample values
        print(f"\nFirst 10 sigma_vM values:")
        print(sigma_vM[:10] / 1e6)
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Stress computed from F and P!")
        print(f"{'='*60}\n")

