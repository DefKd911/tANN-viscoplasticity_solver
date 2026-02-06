#!/usr/bin/env python3
"""
Build baseline ML dataset for U-Net (tCNN) from DAMASK outputs.

X (64,64,5) = [E, nu, xi0, h0, sigma_vM(t)]
Y (64,64,1) = [sigma_vM(t+dt)]

Inputs:
- simulation_results/hdf5_files/seed*.hdf5
- labels/[labels_]seedXXXX.npy           (64x64 ints: grain ids 0..G-1)
- props/[props_]seedXXXX.npy             (G x 4 array: [E, nu, xi0, h0])

Outputs:
ML_DATASET/
  train/inputs/sample_XXXXX.npy
  train/outputs/sample_XXXXX.npy
  val/...
  test/...
  metadata/{seeds_used.txt, normalization.json, increments_map.csv}

Normalization (applied to inputs+target); matches paper Table 3 and "normalized to 1":
- E:   (E-50e9)/250e9
- nu:  (nu-0.2)/0.2
- xi0: (xi0-50e6)/250e6
- h0:  h0/50e9
- sigma_vM: (sigma_vM[Pa]/1e6)/1000  (so MAE in MPa = normalized MAE × 1000)

For paper replication use --max-grains 10 and 80/20 split (e.g. --train 800 --val 200).
"""
import os
import glob
import json
import argparse
import h5py
import numpy as np
from typing import Tuple, List, Optional

HDF5_DIR = "simulation_results/hdf5_files"
LABELS_DIR = "labels"
PROPS_DIR = "props"
OUT_DIR = "ML_DATASET"

# --------------------------- math helpers ---------------------------

def compute_cauchy_from_F_P(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute Cauchy stress from F and P for homogenization fields.
    F: (N,3,3)  P: (N,3,3)
    Returns sigma: (N,3,3)
    """
    if F.ndim != 3 or P.ndim != 3 or F.shape[1:] != (3,3) or P.shape[1:] != (3,3):
        raise ValueError(f"Unexpected shapes F={F.shape}, P={P.shape}")
    N = F.shape[0]
    sigma = np.zeros_like(P)
    for i in range(N):
        Fi = F[i]
        Pi = P[i]
        detF = np.linalg.det(Fi)
        if abs(detF) < 1e-14:
            raise ValueError("det(F) ~ 0 encountered")
        sigma[i] = (1.0/detF) * (Pi @ Fi.T)
    return sigma


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    """Compute von Mises stress field from sigma (N,3,3) -> (N,)."""
    N = sigma.shape[0]
    vm = np.zeros(N, dtype=np.float64)
    I = np.eye(3)
    for i in range(N):
        s = sigma[i]
        s_dev = s - np.trace(s)/3.0 * I
        vm[i] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
    return vm


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


def _reshape_tensor(arr: np.ndarray) -> np.ndarray:
    """Return array shaped (N,3,3) regardless of original grid layout."""
    if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        return arr
    if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    if arr.ndim == 5 and arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    raise ValueError(f"Unexpected tensor shape {arr.shape}")


def _load_dataset(f: h5py.File, base: str, relative_paths: List[str]) -> Optional[np.ndarray]:
    for rel in relative_paths:
        path = f"{base}/{rel}"
        if path in f:
            return f[path][()]
    return None


def load_sigma_tensor(f: h5py.File, base: str) -> Optional[np.ndarray]:
    arr = _load_dataset(f, base, SIGMA_DATASETS)
    if arr is None:
        return None
    return _reshape_tensor(arr)


def load_sigma_from_F_P(f: h5py.File, base: str) -> np.ndarray:
    Farr = _load_dataset(f, base, F_DATASETS)
    Parr = _load_dataset(f, base, P_DATASETS)
    if Farr is None or Parr is None:
        raise KeyError(f"No F/P tensors found under {base}")
    F = _reshape_tensor(Farr)
    P = _reshape_tensor(Parr)
    return compute_cauchy_from_F_P(F, P)


def load_vm_field(f: h5py.File, base: str) -> np.ndarray:
    """Return von Mises field (N,) for a given increment base path."""
    sigma = load_sigma_tensor(f, base)
    if sigma is None:
        sigma = load_sigma_from_F_P(f, base)
    return von_mises_stress(sigma)

    """Compute von Mises stress field from sigma (N,3,3) -> (N,).
    """
    N = sigma.shape[0]
    vm = np.zeros(N, dtype=np.float64)
    I = np.eye(3)
    for i in range(N):
        s = sigma[i]
        s_dev = s - np.trace(s)/3.0 * I
        vm[i] = np.sqrt(1.5 * np.sum(s_dev * s_dev))
    return vm

# --------------------------- io helpers ---------------------------

def find_labels_path(seed_name: str) -> str:
    candidates = [
        os.path.join(LABELS_DIR, f"labels_{seed_name}.npy"),
        os.path.join(LABELS_DIR, f"{seed_name}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Labels file not found for {seed_name} in {LABELS_DIR}")


def find_props_path(seed_name: str) -> str:
    candidates = [
        os.path.join(PROPS_DIR, f"props_{seed_name}.npy"),
        os.path.join(PROPS_DIR, f"{seed_name}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Props file not found for {seed_name} in {PROPS_DIR}")


def get_grain_count(seed_name: str) -> int:
    """Return number of grains for this seed (from props or labels). Used for --max-grains filter."""
    try:
        props_path = find_props_path(seed_name)
        raw = np.load(props_path, allow_pickle=True)
        if getattr(raw, "ndim", 0) == 0 and isinstance(raw.item(), dict):
            d = raw.item()
            return len(np.atleast_1d(d["E"]))
        return int(raw.shape[0])
    except FileNotFoundError:
        pass
    labels_path = find_labels_path(seed_name)
    labels = np.load(labels_path, allow_pickle=True)
    if getattr(labels, "ndim", 0) == 0:
        labels = labels.item()
    return int(len(np.unique(labels)))


def per_pixel_maps(labels: np.ndarray, props: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast per-grain props (Gx4 [E,nu,xi0,h0]) to per-pixel maps using labels (64x64).
    Returns E_map, nu_map, xi0_map, h0_map as (64,64) arrays.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D (H,W)")
    H, W = labels.shape
    # props shape expected (G,4)
    if props.ndim != 2 or props.shape[1] < 4:
        raise ValueError("props must be (G,4) with [E, nu, xi0, h0]")
    # Ensure all grain ids in labels exist in props
    max_label = labels.max()
    if max_label >= props.shape[0]:
        raise ValueError(f"labels contain grain id {max_label} but props only has {props.shape[0]} rows")
    E_map = props[labels, 0]
    nu_map = props[labels, 1]
    xi0_map = props[labels, 2]
    h0_map = props[labels, 3]
    return E_map, nu_map, xi0_map, h0_map

# --------------------------- normalization ---------------------------

def normalize_channels(E_map, nu_map, xi0_map, h0_map, sigma_vm_t, sigma_vm_tp1):
    """Normalize per paper's ranges to ~[0,1]. Input sigma in Pa.
    Returns normalized versions and a dict of ranges.
    """
    # E (50-300 GPa)
    E_norm = (E_map - 50e9) / 250e9
    # nu (0.2-0.4)
    nu_norm = (nu_map - 0.2) / 0.2
    # xi0 (50-300 MPa)
    xi0_norm = (xi0_map - 50e6) / 250e6
    # h0 (0-50 GPa)
    h0_norm = (h0_map) / 50e9
    # sigma vm in MPa scaled by 1000
    sigma_vm_t_norm = (sigma_vm_t / 1e6) / 1000.0
    sigma_vm_tp1_norm = (sigma_vm_tp1 / 1e6) / 1000.0
    return E_norm, nu_norm, xi0_norm, h0_norm, sigma_vm_t_norm, sigma_vm_tp1_norm

# --------------------------- main build ---------------------------

def build_dataset(n_train: int = 800, n_val: int = 100, n_test: int = 0, max_seeds: int = 900, test_hdf5_dir: str = None, max_grains: Optional[int] = None):
    """
    Build ML_DATASET with deterministic seed split.
    Default: use first 900 seeds -> 800 train, 100 val, 0 test (test prepared later).
    If test_hdf5_dir is set, test seeds are read from that folder (separate test HDF5).
    If max_grains is set (e.g. 10 for paper replication), only seeds with that many grains are included.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, split, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "metadata"), exist_ok=True)

    # Train/val: from main HDF5_DIR
    h5_files = sorted(glob.glob(os.path.join(HDF5_DIR, "seed*.hdf5")))
    all_seeds = [os.path.basename(p).replace('.hdf5','') for p in h5_files]
    seeds = all_seeds[:max_seeds] if max_seeds else all_seeds

    if max_grains is not None:
        kept = []
        for s in seeds:
            try:
                if get_grain_count(s) == max_grains:
                    kept.append(s)
            except Exception:
                pass
        n_before = len(seeds)
        seeds = kept
        print(f"[INFO] Filtering to {max_grains}-grain microstructures: {len(seeds)}/{n_before} seeds kept", flush=True)
        if len(seeds) == 0:
            raise ValueError(f"No seeds with exactly {max_grains} grains. Run verify_10grain.py to check.")

    n = len(seeds)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    train_seeds = seeds[:n_train]
    val_seeds = seeds[n_train:n_train + n_val]
    # Test: from test_hdf5_dir if provided, else from main pool
    if test_hdf5_dir and os.path.isdir(test_hdf5_dir):
        test_h5_files = sorted(glob.glob(os.path.join(test_hdf5_dir, "seed*.hdf5")))
        test_seeds = [os.path.basename(p).replace('.hdf5','') for p in test_h5_files]
        if max_grains is not None:
            kept = []
            for s in test_seeds:
                try:
                    if get_grain_count(s) == max_grains:
                        kept.append(s)
                except Exception:
                    pass
            test_seeds = kept
        n_test = len(test_seeds)
        print(f"[INFO] Train/val from {HDF5_DIR}; test from {test_hdf5_dir} ({n_test} seeds)", flush=True)
    else:
        n_test = min(n_test, max(0, n - n_train - n_val))
        test_seeds = seeds[n_train + n_val:n_train + n_val + n_test]
        test_hdf5_dir = None
    print(f"[INFO] train={len(train_seeds)}, val={len(val_seeds)}, test={len(test_seeds)}", flush=True)
    print("[INFO] Building samples (this may take 15-45 min). Progress every 10 seeds...", flush=True)

    with open(os.path.join(OUT_DIR, "metadata", "seeds_used.txt"), "w") as f:
        f.write("TRAIN\n"); [f.write(s+"\n") for s in train_seeds]
        f.write("\nVAL\n"); [f.write(s+"\n") for s in val_seeds]
        f.write("\nTEST\n"); [f.write(s+"\n") for s in test_seeds]

    increments_log = open(os.path.join(OUT_DIR, "metadata", "increments_map.csv"), "w")
    increments_log.write("split,sample_idx,seed,increment_t\n")

    sample_idx = 0

    def process_seed(seed_name: str, split: str, h5_dir: str = None):
        nonlocal sample_idx
        base = h5_dir if h5_dir else HDF5_DIR
        h5_path = os.path.join(base, f"{seed_name}.hdf5")
        labels_path = find_labels_path(seed_name)
        props_path = find_props_path(seed_name)

        labels = np.load(labels_path, allow_pickle=True)  # (64,64)
        if getattr(labels, 'ndim', 0) == 0:
            labels = labels.item()
        props_raw = np.load(props_path, allow_pickle=True)    # dict or (G,4)
        if getattr(props_raw, 'ndim', 0) == 0 and isinstance(props_raw.item(), dict):
            d = props_raw.item()
            # Expect arrays of shape (G,)
            E_arr = np.asarray(d['E'])
            nu_arr = np.asarray(d['nu'])
            xi0_arr = np.asarray(d['xi0'])
            h0_arr = np.asarray(d['h0'])
            props = np.stack([E_arr, nu_arr, xi0_arr, h0_arr], axis=1)
        else:
            props = props_raw
        E_map, nu_map, xi0_map, h0_map = per_pixel_maps(labels, props)

        with h5py.File(h5_path, 'r') as f:
            # enumerate increments present
            incs = sorted([k for k in f.keys() if k.startswith('increment_')], key=lambda x: int(x.split('_')[1]))
            # produce pairs (t, t+1)
            for ii in range(len(incs)-1):
                t_name = incs[ii]; tp1_name = incs[ii+1]
                base_t = f"{t_name}/homogenization/h0"
                base_tp1 = f"{tp1_name}/homogenization/h0"
                try:
                    vm_t = load_vm_field(f, base_t)
                    vm_tp1 = load_vm_field(f, base_tp1)
                except KeyError as err:
                    print(f"[WARN] Missing tensors for {seed_name} {t_name}->{tp1_name}: {err}")
                    continue

                # reshape to (64,64)
                H = W = 64
                if vm_t.shape[0] != H*W:
                    raise ValueError(f"Unexpected field size {vm_t.shape[0]}, expected {H*W}")
                vm_t_2d = vm_t.reshape(H, W)
                vm_tp1_2d = vm_tp1.reshape(H, W)

                # normalize channels
                E_n, nu_n, xi0_n, h0_n, vm_t_n, vm_tp1_n = normalize_channels(
                    E_map, nu_map, xi0_map, h0_map, vm_t_2d, vm_tp1_2d
                )

                # stack inputs/outputs
                X = np.stack([E_n, nu_n, xi0_n, h0_n, vm_t_n], axis=-1).astype(np.float32)  # (64,64,5)
                Y = vm_tp1_n[..., None].astype(np.float32)                                  # (64,64,1)

                # save
                split_dir_in = os.path.join(OUT_DIR, split, "inputs")
                split_dir_out = os.path.join(OUT_DIR, split, "outputs")
                in_path = os.path.join(split_dir_in, f"sample_{sample_idx:05d}.npy")
                out_path = os.path.join(split_dir_out, f"sample_{sample_idx:05d}.npy")
                np.save(in_path, X)
                np.save(out_path, Y)
                increments_log.write(f"{split},{sample_idx},{seed_name},{ii}\n")
                sample_idx += 1

    # process splits (with progress: 900 seeds can take 15–45 min)
    for i, s in enumerate(train_seeds):
        process_seed(s, "train")
        if (i + 1) % 10 == 0 or (i + 1) == len(train_seeds):
            print(f"[INFO] Train: {i+1}/{len(train_seeds)} seeds -> {sample_idx} samples", flush=True)
    for i, s in enumerate(val_seeds):
        process_seed(s, "val")
        if (i + 1) % 10 == 0 or (i + 1) == len(val_seeds):
            print(f"[INFO] Val: {i+1}/{len(val_seeds)} seeds -> {sample_idx} samples", flush=True)
    for i, s in enumerate(test_seeds):
        process_seed(s, "test", h5_dir=test_hdf5_dir)
        if test_hdf5_dir and ((i + 1) % 10 == 0 or (i + 1) == len(test_seeds)):
            print(f"[INFO] Test: {i+1}/{len(test_seeds)} seeds -> {sample_idx} samples", flush=True)

    increments_log.close()

    # write normalization used
    norm_info = {
        "E": {"min": 50e9, "max": 300e9, "formula": "(E-50e9)/250e9"},
        "nu": {"min": 0.2, "max": 0.4, "formula": "(nu-0.2)/0.2"},
        "xi0": {"min": 50e6, "max": 300e6, "formula": "(xi0-50e6)/250e6"},
        "h0": {"min": 0.0, "max": 50e9, "formula": "h0/50e9"},
        "sigma_vM": {"scale": "MPa/1000", "formula": "(sigma_vM[Pa]/1e6)/1000"}
    }
    with open(os.path.join(OUT_DIR, "metadata", "normalization.json"), "w") as f:
        json.dump(norm_info, f, indent=2)

    print("\n============================================")
    print("[SUCCESS] ML_DATASET generated at ./ML_DATASET")
    print(f"  - Total samples: {sample_idx}")
    print("  - Splits: train/val/test (inputs 64x64x5, outputs 64x64x1)")
    print("  - Metadata: seeds_used.txt, normalization.json, increments_map.csv")
    print("============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML dataset from DAMASK HDF5 (800 train, 100 val, test later).")
    parser.add_argument("--train", type=int, default=800, help="Number of seeds for training")
    parser.add_argument("--val", type=int, default=100, help="Number of seeds for validation")
    parser.add_argument("--test", type=int, default=0, help="Number of seeds for test (0 = prepare later)")
    parser.add_argument("--max-seeds", type=int, default=900, help="Max seeds to use (first N); 0 = use all")
    parser.add_argument("--test-hdf5-dir", type=str, default=None, help="Folder with test HDF5 only (e.g. simulation_results/test_hdf5_files)")
    parser.add_argument("--max-grains", type=int, default=None, help="Only include seeds with this many grains (e.g. 10 for paper replication)")
    args = parser.parse_args()
    build_dataset(n_train=args.train, n_val=args.val, n_test=args.test, max_seeds=args.max_seeds, test_hdf5_dir=args.test_hdf5_dir, max_grains=args.max_grains)
