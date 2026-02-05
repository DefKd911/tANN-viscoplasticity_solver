#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt


def vm_from_sigma(sigma: np.ndarray) -> np.ndarray:
    tr = np.trace(sigma, axis1=1, axis2=2)
    s_dev = sigma - tr[:, None, None] / 3.0 * np.eye(3)[None, :, :]
    return np.sqrt(1.5 * np.sum(s_dev**2, axis=(1, 2)))


def sigma_from_FS(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    J = np.linalg.det(F)
    return np.einsum('...ik,...kl,...jl->...ij', F, S, F) / J[:, None, None]


def main():
    parser = argparse.ArgumentParser(description='Build stress–strain curve from DAMASK HDF5')
    parser.add_argument('--hdf5', required=True, help='Path to HDF5 with increments')
    parser.add_argument('--output', default='ss_curve.png', help='Output plot')
    args = parser.parse_args()

    with h5py.File(args.hdf5, 'r') as f:
        incs = sorted([k for k in f.keys() if k.startswith('increment_')], key=lambda s: int(s.split('_')[1]))
        if not incs:
            raise RuntimeError('No increments found in HDF5')

        strains, sigmas = [], []
        # Try to compute strain from solver.F (cell-wise) or geometry u_p as fallback
        # Prefer solver.F_lastInc or per-increment geometry if available
        # We approximate macro engineering strain as mean(F11) - 1.
        for inc in incs:
            g = f[inc]
            # Macro strain
            if 'geometry' in g and 'u_p' in g['geometry'] and 'size' in f.get('geometry', {}):
                up = g['geometry']['u_p'][:]
                Lx = float(f['geometry']['size'][0]) if 'geometry' in f else 1.0
                exx = np.mean(up[:, 0]) / Lx
            elif 'solver' in f and 'F_lastInc' in f['solver']:
                Fcells = f['solver']['F_lastInc'][:]
                exx = np.mean(Fcells[:, 0, 0]) - 1.0
            else:
                # Attempt per-increment field F in phase (rare)
                exx = np.nan
            strains.append(exx)

            # Macro stress sigma11
            sigma11 = None
            # Check phase stress_Cauchy first (most common with pass homogenization)
            if 'phase' in g:
                sigma_cells = []
                for pk, pg in g['phase'].items():
                    if 'mechanical' in pg and 'stress_Cauchy' in pg['mechanical']:
                        s = pg['mechanical']['stress_Cauchy'][:]
                        if s.ndim == 3:
                            sigma_cells.append(s[:, 0, 0])
                        elif s.ndim == 2 and s.shape[1] == 6:
                            sigma_cells.append(s[:, 0])
                if sigma_cells:
                    sigma11 = float(np.mean(np.concatenate(sigma_cells)))
            
            # Fallback: homogenization stress_Cauchy if present
            if sigma11 is None and 'homogenization' in g:
                for hk, hg in g['homogenization'].items():
                    if 'mechanical' in hg and 'stress_Cauchy' in hg['mechanical']:
                        s = hg['mechanical']['stress_Cauchy'][:]
                        if s.ndim == 3:
                            sigma11 = float(np.mean(s[:, 0, 0]))
                        elif s.ndim == 2 and s.shape[1] == 6:
                            sigma11 = float(np.mean(s[:, 0]))
                        break
            
            # Last resort: reconstruct from F,S
            if sigma11 is None and 'phase' in g:
                # Reconstruct from F,S if available per phase
                sigma_cells = []
                for pk, pg in g['phase'].items():
                    if 'F' in pg and 'S' in pg:
                        Fp = pg['F'][:]
                        Sp = pg['S'][:]
                        scells = sigma_from_FS(Fp, Sp)
                        sigma_cells.append(scells)
                if sigma_cells:
                    s_all = np.concatenate(sigma_cells, axis=0)
                    sigma11 = float(np.mean(s_all[:, 0, 0]))
            if sigma11 is None:
                raise RuntimeError(f'No stress data found in {inc} (stress_Cauchy or F,S).')
            sigmas.append(sigma11)

    strains = np.array(strains)
    sigmas_mpa = np.array(sigmas) / 1e6

    plt.figure()
    plt.plot(strains, sigmas_mpa, '-o', lw=2)
    plt.xlabel('Engineering strain $\\epsilon_{xx}$')
    plt.ylabel('$\\sigma_{xx}$ [MPa]')
    plt.title('Stress–strain curve')
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f'[SUCCESS] Saved {args.output}')

if __name__ == '__main__':
    main()
