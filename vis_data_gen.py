# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# ------------------------------
# Data classes for configuration
# ------------------------------

@dataclass
class PropertyRanges:
    """Material property ranges (inclusive of endpoints)."""
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
    """Helper to create a numpy Generator with optional seed."""
    return np.random.default_rng(seed)


def sample_grain_properties(K: int, pr: PropertyRanges, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Sample per-grain properties from uniform ranges.
    Returns a dict of arrays with shape (K,).
    
    Units:
      - E, h0 returned in Pascals (input in GPa).
      - xi0 returned in Pascals (input in MPa).
      - nu is dimensionless.
    """
    E = rng.uniform(pr.E_min_GPa, pr.E_max_GPa, size=K) * 1e9      # Pa
    nu = rng.uniform(pr.nu_min, pr.nu_max, size=K)                  # -
    xi0 = rng.uniform(pr.xi0_min_MPa, pr.xi0_max_MPa, size=K) * 1e6 # Pa
    h0 = rng.uniform(pr.h0_min_GPa, pr.h0_max_GPa, size=K) * 1e9    # Pa
    return {"E": E, "nu": nu, "xi0": xi0, "h0": h0}


def generate_voronoi_labels(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a Voronoi-style partition on an n x n grid by nearest-center assignment.
    This yields a discrete Voronoi tessellation without explicitly computing polygons.
    
    Returns:
        labels: (n, n) integer array with values in [0, K-1], grain id per pixel.
    """
    # Random centers in the unit square
    centers = rng.uniform(0, 1, size=(K, 2))
    
    # Grid of coordinates in [0,1] x [0,1]
    xs = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    ys = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # shape (n, n)
    grid = np.stack([X, Y], axis=-1)           # (n, n, 2)
    
    # Compute squared distances to each center: (n, n, K)
    # Using broadcasting: (n, n, 2) - (K, 2) -> (n, n, K, 2) -> sum over last axis
    diffs = grid[:, :, None, :] - centers[None, None, :, :]
    d2 = np.sum(diffs**2, axis=-1)
    labels = np.argmin(d2, axis=-1).astype(np.int32)
    return labels


def rasterize_properties(labels: np.ndarray, grain_props: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Rasterize per-grain properties to per-pixel property maps.
    
    Args:
        labels: (n, n) integer grain labels
        grain_props: dict with keys "E", "nu", "xi0", "h0", each (K,)
    Returns:
        dict of 2D arrays (n, n) for each property
    """
    prop_maps = {}
    for key, vec in grain_props.items():
        prop_maps[key] = vec[labels]
    return prop_maps


def build_input_tensor(prop_maps: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Build a 5-channel tensor (n, n, 5) in the order:
        [E, nu, xi0, h0, von_mises_init]
    where von_mises_init is initialized to zeros.
    """
    n = prop_maps["E"].shape[0]
    
    tensor = np.stack(
        [
            prop_maps["E"].astype(np.float32),
            prop_maps["nu"].astype(np.float32),
            prop_maps["xi0"].astype(np.float32),
            prop_maps["h0"].astype(np.float32),
            
        ],
        axis=-1,
    )
    return tensor


def generate_microstructure(
    n: int = 64,
    K: int = 10,
    prop_ranges: PropertyRanges = PropertyRanges(),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Generate a single microstructure:
      - Voronoi labels (n, n)
      - Property maps dict of 2D arrays: E (Pa), nu (-), xi0 (Pa), h0 (Pa)
      - 5-channel input tensor (n, n, 5): [E, nu, xi0, h0, von_mises_init]
    """
    rng = _rng(seed)
    labels = generate_voronoi_labels(n, K, rng)
    grain_props = sample_grain_properties(K, prop_ranges, rng)
    prop_maps = rasterize_properties(labels, grain_props)
    tensor = build_input_tensor(prop_maps)
    return labels, prop_maps, tensor


def generate_dataset(
    N: int,
    n: int = 64,
    K: int = 10,
    prop_ranges: PropertyRanges = PropertyRanges(),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a dataset of N microstructures as a numpy array of shape (N, n, n, 5).
    """
    rng_master = _rng(seed)
    tensors = []
    for i in range(N):
        # Derive a seed per sample for reproducibility
        sample_seed = int(rng_master.integers(0, 2**31 - 1))
        _, _, tensor = generate_microstructure(n=n, K=K, prop_ranges=prop_ranges, seed=sample_seed)
        tensors.append(tensor)
    return np.stack(tensors, axis=0)


# ------------------------------
# Visualization helpers
# ------------------------------

def show_microstructure(labels: np.ndarray) -> None:
    """Visualize grain labels."""
    plt.figure()
    plt.imshow(labels, interpolation='nearest')
    plt.title("Voronoi Grain Map (labels)")
    plt.axis('off')
    plt.show()


def show_property_map(name: str, arr: np.ndarray, unit: str) -> None:
    """Visualize a single property map with a colorbar."""
    plt.figure()
    im = plt.imshow(arr, interpolation='nearest')
    plt.title(f"{name} [{unit}]")
    plt.colorbar(im)
    plt.axis('off')
    plt.show()


def demo_visualization(seed: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Generate one sample and display:
      - grain labels
      - E (GPa), nu (-), xi0 (MPa), h0 (GPa)
      - von Mises initial (Pa, zeros)
    Returns the labels, property maaps, and input tensor.
    """
    labels, props, tensor = generate_microstructure(seed=seed)
    np.save(f"labels_seed{seed}.npy", labels)
    show_microstructure(labels)
    
    # Display in friendly units for visualization
    show_property_map("Young's modulus E", props["E"] / 1e9, "GPa")
    show_property_map("Poisson's ratio v", props["nu"], "-")
    show_property_map("Initial flow resistance ξ0", props["xi0"] / 1e6, "MPa")
    show_property_map("Linear isotropic hardening h0", props["h0"] / 1e9, "GPa")
    return labels, props, tensor


# ------------------------------
# Example usage (run once)
# ------------------------------

if __name__ == "__main__":
    # Generate and visualize a single microstructure
    labels, props, tensor = demo_visualization(seed=123)
    
    # Generate a small dataset (e.g., 4 samples)
    dataset = generate_dataset(N=10, seed=2115)
    print("Dataset shape:", dataset.shape)


# %%
