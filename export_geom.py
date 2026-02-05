# export_geom.py
# Convert saved Voronoi labels (.npy) into DAMASK-compatible geometry (.vti)

import numpy as np
import damask
from typing import Tuple, Sequence

def _renumber_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renumber labels to consecutive integers starting at 0.
    Returns: (renumbered_labels, mapping_array)
    """
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    vec_map = np.vectorize(lambda x: remap[x], otypes=[np.int32])
    labels_ren = vec_map(labels).astype(np.int32)

    max_old = int(unique.max())
    mapping = -1 * np.ones((max_old + 1,), dtype=np.int32)
    for old in unique:
        mapping[int(old)] = remap[int(old)]
    return labels_ren, mapping

def labels_to_geom(
    labels: np.ndarray,
    size: Sequence[float] = (1e-5, 1e-5, 1e-6),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    out_vti: str = "micro.vti",
    compress: bool = True,
) -> damask.GeomGrid:
    """
    Convert a 2D labels array (ny, nx) to a DAMASK GeomGrid and save as .vti.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array (ny, nx)")

    labels_ren, mapping = _renumber_labels(labels)

    # DAMASK GeomGrid expects 3D material grid → add 1 z-layer
    material_3d = labels_ren[:, :, None].astype(np.int32)

    size_arr = np.asarray(size, dtype=float)
    origin_arr = np.asarray(origin, dtype=float)

    g = damask.GeomGrid(material=material_3d, size=size_arr, origin=origin_arr)
    g.save(out_vti, compress=compress)
    print(f"[export_geom] wrote '{out_vti}'  (cells: {g.cells}, #materials: {g.N_materials})")
    return g

# ------------------------------
# CLI entry point
# ------------------------------
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Export DAMASK GeomGrid from labels.npy")
    parser.add_argument("--labels-npy", type=str, required=True, help="path to labels .npy file")
    parser.add_argument("--out", type=str, default="micro.vti", help="output .vti filename")
    args = parser.parse_args()

    if not os.path.exists(args.labels_npy):
        raise FileNotFoundError(f"Labels file not found: {args.labels_npy}")

    labels = np.load(args.labels_npy)
    labels_to_geom(labels, out_vti=args.out)
