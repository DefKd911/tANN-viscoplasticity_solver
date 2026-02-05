# train_data_gen.py
# Generate synthetic microstructures (labels + properties) for DAMASK + ML

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# ------------------------------
# Config for property sampling
# ------------------------------

@dataclass
class PropertyRanges:
    """Material property ranges (inclusive)."""
    E_min_GPa: float = 50.0
    E_max_GPa: float = 300.0
    nu_min: float = 0.2
    nu_max: float = 0.4
    xi0_min_MPa: float = 50.0
    xi0_max_MPa: float = 300.0
    h0_min_GPa: float = 0.0
    h0_max_GPa: float = 50.0

# ------------------------------
# Core generation utilities
# ------------------------------

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def sample_grain_properties(K: int, pr: PropertyRanges, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Sample per-grain properties (E, nu, xi0, h0)."""
    E   = rng.uniform(pr.E_min_GPa,  pr.E_max_GPa,  size=K) * 1e9
    nu  = rng.uniform(pr.nu_min,     pr.nu_max,     size=K)
    xi0 = rng.uniform(pr.xi0_min_MPa, pr.xi0_max_MPa, size=K) * 1e6
    h0  = rng.uniform(pr.h0_min_GPa, pr.h0_max_GPa, size=K) * 1e9
    return {"E": E, "nu": nu, "xi0": xi0, "h0": h0}

def generate_voronoi_labels(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Voronoi tessellation labels (n x n)."""
    centers = rng.uniform(0, 1, size=(K, 2))
    xs = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    ys = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([X, Y], axis=-1)
    d2 = np.sum((grid[:, :, None, :] - centers[None, None, :, :])**2, axis=-1)
    return np.argmin(d2, axis=-1).astype(np.int32)

def generate_microstructure(n: int = 64, K: int = 10,
                            prop_ranges: PropertyRanges = PropertyRanges(),
                            seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate labels + grain properties."""
    rng = _rng(seed)
    labels = generate_voronoi_labels(n, K, rng)
    grain_props = sample_grain_properties(K, prop_ranges, rng)
    return labels, grain_props

# ------------------------------
# Example usage
# ------------------------------

if __name__ == "__main__":
    seed = 123
    labels, props = generate_microstructure(seed=seed)

    # Save to .npy for later DAMASK export
    np.save(f"labels_seed{seed}.npy", labels)
    np.save(f"props_seed{seed}.npy", props, allow_pickle=True)

    print(f"[train_data_gen] Saved labels_seed{seed}.npy and props_seed{seed}.npy")
