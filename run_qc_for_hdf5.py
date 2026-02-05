#!/usr/bin/env python3
"""
run_qc_for_hdf5.py

Convenience wrapper that:
  1. Iterates over available simulation_results/hdf5_files/seed*.hdf5 (or a user-specified list)
  2. Runs plot_ss_curve_from_F_P.py on each file (skips if inputs missing)
  3. After all successful stress-curve generations, calls visualize_gt_from_hdf5.py
     for the same subset to produce 7-panel GT overlays with elastic checks.

Usage:
    python run_qc_for_hdf5.py
    python run_qc_for_hdf5.py --seeds 1001763537,1003283475
"""

import argparse
import glob
import os
import subprocess
import sys


HDF5_DIR = os.path.join("simulation_results", "hdf5_files")
STRESS_CURVE_SCRIPT = "plot_ss_curve_from_F_P.py"
VIS_SCRIPT = "visualize_gt_from_hdf5.py"


def run_cmd(cmd, env=None):
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="", help="Comma-separated list of seeds. Default: all in hdf5 dir.")
    ap.add_argument("--hdf5-dir", default=HDF5_DIR)
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--props-dir", default="props")
    ap.add_argument("--out", default=os.path.join("ML_EVAL", "gt_overlays"))
    args = ap.parse_args()

    if args.seeds.strip():
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [
            os.path.splitext(os.path.basename(p))[0].replace("seed", "")
            for p in sorted(glob.glob(os.path.join(args.hdf5_dir, "seed*.hdf5")))
        ]

    if not seeds:
        print("[WARN] No seeds found to process.")
        return

    py = sys.executable
    processed = []

    for seed in seeds:
        h5_path = os.path.join(args.hdf5_dir, f"seed{seed}.hdf5")
        if not os.path.isfile(h5_path):
            print(f"[SKIP] Missing {h5_path}")
            continue
        print(f"\n[RUN] Stress curve for seed{seed}")
        ret = run_cmd([py, STRESS_CURVE_SCRIPT, h5_path])
        if ret == 0:
            processed.append(seed)
        else:
            print(f"[WARN] Stress curve script failed for seed{seed}; skipping visualization.")

    if not processed:
        print("[INFO] No successful stress curves → nothing to visualize.")
        return

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    vis_cmd = [
        py,
        VIS_SCRIPT,
        "--hdf5-dir",
        args.hdf5_dir,
        "--labels-dir",
        args.labels_dir,
        "--props-dir",
        args.props_dir,
        "--seeds",
        ",".join(processed),
        "--out",
        args.out,
        "--overlay-boundaries",
        "--auto-align",
        "--elastic-check",
    ]
    print(f"\n[RUN] Visualization for {len(processed)} seeds")
    run_cmd(vis_cmd, env=env)


if __name__ == "__main__":
    main()

