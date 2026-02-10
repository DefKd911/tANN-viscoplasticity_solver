#!/usr/bin/env python3
"""
Build only the test split of ML_DATASET from a separate HDF5 folder (e.g. test set).
Does not touch train/val. Uses same normalization as build_ml_dataset.py.

Usage:
  python build_test_split.py --test-hdf5-dir simulation_results/test_hdf5_files
  python build_test_split.py --test-hdf5-dir simulation_results/test_hdf5_files --dataset ML_DATASET
"""
import os
import glob
import argparse
import h5py
import numpy as np

# Reuse helpers from build_ml_dataset (same normalization and I/O)
from build_ml_dataset import (
    load_vm_field,
    normalize_channels,
    find_labels_path,
    find_props_path,
    per_pixel_maps,
    OUT_DIR,
)

def build_test_split(test_hdf5_dir: str, dataset_dir: str = None):
    out_dir = dataset_dir or OUT_DIR
    test_in = os.path.join(out_dir, "test", "inputs")
    test_out = os.path.join(out_dir, "test", "outputs")
    meta_dir = os.path.join(out_dir, "metadata")
    os.makedirs(test_in, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(test_hdf5_dir, "seed*.hdf5")))
    test_seeds = [os.path.basename(p).replace(".hdf5", "") for p in h5_files]
    if not test_seeds:
        print(f"[WARN] No seed*.hdf5 found in {test_hdf5_dir}")
        return

    # Clear existing test samples so we don't mix old/new
    for d in [test_in, test_out]:
        for f in glob.glob(os.path.join(d, "sample_*.npy")):
            os.remove(f)

    sample_idx = 0
    increments_rows = []

    for i, seed_name in enumerate(test_seeds):
        h5_path = os.path.join(test_hdf5_dir, f"{seed_name}.hdf5")
        try:
            labels = np.load(find_labels_path(seed_name), allow_pickle=True)
            if getattr(labels, "ndim", 0) == 0:
                labels = labels.item()
            props_raw = np.load(find_props_path(seed_name), allow_pickle=True)
            if getattr(props_raw, "ndim", 0) == 0 and isinstance(props_raw.item(), dict):
                d = props_raw.item()
                props = np.stack([np.asarray(d["E"]), np.asarray(d["nu"]), np.asarray(d["xi0"]), np.asarray(d["h0"])], axis=1)
            else:
                props = props_raw
        except Exception as e:
            print(f"[WARN] Skip {seed_name}: {e}")
            continue

        E_map, nu_map, xi0_map, h0_map = per_pixel_maps(labels, props)

        with h5py.File(h5_path, "r") as f:
            incs = sorted([k for k in f.keys() if k.startswith("increment_")], key=lambda x: int(x.split("_")[1]))
            for ii in range(len(incs) - 1):
                t_name, tp1_name = incs[ii], incs[ii + 1]
                base_t = f"{t_name}/homogenization/h0"
                base_tp1 = f"{tp1_name}/homogenization/h0"
                try:
                    vm_t = load_vm_field(f, base_t)
                    vm_tp1 = load_vm_field(f, base_tp1)
                except KeyError as err:
                    print(f"[WARN] {seed_name} {t_name}->{tp1_name}: {err}")
                    continue
                H = W = 64
                if vm_t.shape[0] != H * W:
                    continue
                vm_t_2d = vm_t.reshape(H, W)
                vm_tp1_2d = vm_tp1.reshape(H, W)
                E_n, nu_n, xi0_n, h0_n, vm_t_n, vm_tp1_n = normalize_channels(
                    E_map, nu_map, xi0_map, h0_map, vm_t_2d, vm_tp1_2d
                )
                X = np.stack([E_n, nu_n, xi0_n, h0_n, vm_t_n], axis=-1).astype(np.float32)
                Y = vm_tp1_n[..., None].astype(np.float32)
                in_path = os.path.join(test_in, f"sample_{sample_idx:05d}.npy")
                out_path = os.path.join(test_out, f"sample_{sample_idx:05d}.npy")
                np.save(in_path, X)
                np.save(out_path, Y)
                increments_rows.append((sample_idx, seed_name, ii))
                sample_idx += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(test_seeds):
            print(f"[INFO] Test: {i+1}/{len(test_seeds)} seeds -> {sample_idx} samples", flush=True)

    # Update metadata: keep TRAIN/VAL, set TEST seeds; keep train/val rows in increments_map, append test rows
    seeds_used_path = os.path.join(meta_dir, "seeds_used.txt")
    inc_map_path = os.path.join(meta_dir, "increments_map.csv")
    train_val_lines = []
    if os.path.exists(seeds_used_path):
        with open(seeds_used_path) as f:
            for line in f:
                if line.strip() == "TEST":
                    break
                train_val_lines.append(line)
    with open(seeds_used_path, "w") as f:
        f.writelines(train_val_lines)
        f.write("\nTEST\n")
        for s in test_seeds:
            f.write(s + "\n")

    existing_non_test = []
    if os.path.exists(inc_map_path):
        with open(inc_map_path) as f:
            for line in f:
                if line.startswith("split,"):
                    continue
                if line.strip().startswith("train,") or line.strip().startswith("val,"):
                    existing_non_test.append(line)
    with open(inc_map_path, "w") as f:
        f.write("split,sample_idx,seed,increment_t\n")
        f.writelines(existing_non_test)
        for idx, seed_name, inc_t in increments_rows:
            f.write(f"test,{idx},{seed_name},{inc_t}\n")

    print(f"[SUCCESS] Test split: {sample_idx} samples from {len(test_seeds)} seeds in {out_dir}/test")
    print(f"  MAE in MPa = normalized MAE × 1000")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build only ML_DATASET/test from a separate HDF5 folder.")
    ap.add_argument("--test-hdf5-dir", default="simulation_results/test_hdf5_files", help="Folder with test seed*.hdf5")
    ap.add_argument("--dataset", default="ML_DATASET", help="Dataset root (default ML_DATASET)")
    args = ap.parse_args()
    build_test_split(args.test_hdf5_dir, args.dataset)
