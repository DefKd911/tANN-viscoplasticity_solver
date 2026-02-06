#!/usr/bin/env python3
"""
Verify how many grains each microstructure has. Paper (Khorrami et al. 2023) uses
10-grain microstructures only. Use this to confirm your data matches, then build
ML_DATASET with --max-grains 10 for paper replication.

Usage:
  python verify_10grain.py
  python verify_10grain.py --hdf5-dir simulation_results/hdf5_files -o seeds_10grain.txt
"""
import os
import glob
import argparse
import numpy as np

LABELS_DIR = "labels"
PROPS_DIR = "props"


def find_labels_path(seed_name: str) -> str:
    for name in [f"labels_{seed_name}.npy", f"{seed_name}.npy"]:
        p = os.path.join(LABELS_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No labels for {seed_name}")


def find_props_path(seed_name: str) -> str:
    for name in [f"props_{seed_name}.npy", f"{seed_name}.npy"]:
        p = os.path.join(PROPS_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No props for {seed_name}")


def get_grain_count(seed_name: str) -> int:
    """Return number of grains for this seed (from props or labels)."""
    # Prefer props: G = number of rows
    try:
        p = find_props_path(seed_name)
        raw = np.load(p, allow_pickle=True)
        if getattr(raw, "ndim", 0) == 0 and isinstance(raw.item(), dict):
            d = raw.item()
            return len(np.atleast_1d(d["E"]))
        return int(raw.shape[0])
    except FileNotFoundError:
        pass
    # Fallback: labels unique ids
    p = find_labels_path(seed_name)
    labels = np.load(p, allow_pickle=True)
    if getattr(labels, "ndim", 0) == 0:
        labels = labels.item()
    return int(len(np.unique(labels)))


def main():
    ap = argparse.ArgumentParser(description="Verify grain count per seed (paper uses 10-grain).")
    ap.add_argument("--hdf5-dir", default="simulation_results/hdf5_files", help="Directory with seed*.hdf5")
    ap.add_argument("--seeds-file", default=None, help="Optional: list of seed names, one per line")
    ap.add_argument("-o", "--out", default=None, help="Write seeds with exactly 10 grains to this file")
    ap.add_argument("--target", type=int, default=10, help="Target grain count (default 10)")
    args = ap.parse_args()

    if args.seeds_file and os.path.exists(args.seeds_file):
        with open(args.seeds_file) as f:
            seeds = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        pattern = os.path.join(args.hdf5_dir, "seed*.hdf5")
        files = sorted(glob.glob(pattern))
        seeds = [os.path.basename(p).replace(".hdf5", "") for p in files]

    if not seeds:
        print(f"No seeds found (hdf5-dir={args.hdf5_dir}, seeds-file={args.seeds_file})")
        return

    counts = {}
    for s in seeds:
        try:
            counts[s] = get_grain_count(s)
        except Exception as e:
            counts[s] = None
            print(f"  Error {s}: {e}")

    n_total = len(seeds)
    n_10 = sum(1 for c in counts.values() if c == args.target)
    n_other = sum(1 for c in counts.values() if c is not None and c != args.target)
    n_fail = sum(1 for c in counts.values() if c is None)

    print(f"Seeds checked: {n_total}")
    print(f"  With exactly {args.target} grains: {n_10}")
    print(f"  With other grain count: {n_other}")
    print(f"  Failed to load: {n_fail}")

    if n_other > 0:
        from collections import Counter
        other_counts = Counter(c for c in counts.values() if c is not None and c != args.target)
        print(f"  Other grain counts: {dict(other_counts)}")
        # Show first few examples
        examples = [s for s, c in counts.items() if c is not None and c != args.target][:5]
        print(f"  Examples (not {args.target}-grain): {examples}")

    if args.out and n_10 > 0:
        ten_grains = [s for s, c in counts.items() if c == args.target]
        with open(args.out, "w") as f:
            for s in ten_grains:
                f.write(s + "\n")
        print(f"Wrote {len(ten_grains)} seeds with {args.target} grains to {args.out}")

    if n_10 == n_total and n_fail == 0:
        print("All microstructures are 10-grain. You can use ML_DATASET as-is for paper replication.")
    elif n_10 > 0:
        print(f"For paper replication, build dataset with --max-grains {args.target} (after creating {args.out or 'a list of 10-grain seeds'}).")


if __name__ == "__main__":
    main()
