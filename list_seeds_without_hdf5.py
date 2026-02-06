#!/usr/bin/env python3
"""
List seeds that have geometry (and material) but no HDF5 yet.
Use these for test-set simulation only.

Usage:
  python list_seeds_without_hdf5.py
  python list_seeds_without_hdf5.py -o test_seeds_to_simulate.txt

Output: prints seed names and optionally writes one seed per line to a file.
"""
import os
import glob
import argparse

GEOM_DIR = "geom_vti"
HDF5_DIR = "simulation_results/hdf5_files"
MATERIAL_DIR = "material_yaml_fixed"

def main():
    parser = argparse.ArgumentParser(description="List seeds that need HDF5 (for test simulation).")
    parser.add_argument("-o", "--out", default=None, help="Write seed list to this file (one per line)")
    args = parser.parse_args()

    geoms = sorted(glob.glob(os.path.join(GEOM_DIR, "run_*.vti")))
    h5s = sorted(glob.glob(os.path.join(HDF5_DIR, "seed*.hdf5")))
    geom_seeds = [os.path.basename(p).replace("run_", "").replace(".vti", "") for p in geoms]
    hdf5_seeds = set(os.path.basename(p).replace(".hdf5", "") for p in h5s)
    missing = sorted(s for s in geom_seeds if s not in hdf5_seeds)

    # Only include seeds that have material file
    with_material = []
    for s in missing:
        if os.path.exists(os.path.join(MATERIAL_DIR, f"material_{s}.yaml")):
            with_material.append(s)

    print(f"[INFO] Total with geometry: {len(geom_seeds)}")
    print(f"[INFO] Total with HDF5: {len(hdf5_seeds)}")
    print(f"[INFO] Missing HDF5 (have geom): {len(missing)}")
    print(f"[INFO] Missing HDF5 and have material (ready to simulate): {len(with_material)}")
    if with_material:
        print(f"[INFO] First 5: {with_material[:5]}")
        print(f"[INFO] Last 5: {with_material[-5:]}")

    if args.out and with_material:
        with open(args.out, "w") as f:
            for s in with_material:
                f.write(s + "\n")
        print(f"[INFO] Wrote {len(with_material)} seeds to {args.out}")
        print(f"[INFO] Run: python batch_run_damask.py --seeds-file {args.out}")

if __name__ == "__main__":
    main()
