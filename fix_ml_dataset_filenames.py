#!/usr/bin/env python3
"""
Fix duplicate / suffixed sample filenames in ML_DATASET so that
train_unet_baseline.py can load matching input/output pairs.

Typical problem (from Windows / gdown re-downloads):
  - ML_DATASET/train/inputs/sample_00736(1).npy
  - ML_DATASET/train/outputs/sample_00736.npy

This script will, for each split (train/val/test) and for both
inputs/ and outputs/:
  - If it sees sample_XXXXX(1).npy and sample_XXXXX.npy already exists,
    it treats the suffixed file as a duplicate and DELETES it.
  - If it sees sample_XXXXX(1).npy and sample_XXXXX.npy does NOT exist,
    it RENAMES sample_XXXXX(1).npy -> sample_XXXXX.npy.

Afterwards it prints a short summary and also checks for any remaining
input files without matching outputs (and vice versa).

Usage (from repo root):
  python fix_ml_dataset_filenames.py
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path("ML_DATASET")
SPLITS = ["train", "val", "test"]

SUFFIX_RE = re.compile(r"^(sample_\d+)(\(\d+\))?\.npy$")


def collect_files(dir_path: Path) -> Dict[str, List[Path]]:
    """
    Return mapping base_name -> list of files for that base in dir_path.
    base_name is the filename without any '(1)', '(2)' suffix, e.g.
      'sample_00736.npy'
    """
    mapping: Dict[str, List[Path]] = {}
    for p in dir_path.glob("*.npy"):
        m = SUFFIX_RE.match(p.name)
        if not m:
            # unexpected pattern; keep as-is using full name as base
            base = p.name
        else:
            base_core = m.group(1)  # e.g. 'sample_00736'
            base = f"{base_core}.npy"
        mapping.setdefault(base, []).append(p)
    return mapping


def fix_dir(dir_path: Path) -> Tuple[int, int]:
    """
    Fix filenames in a single directory (e.g. ML_DATASET/train/inputs).
    Returns (n_renamed, n_deleted).
    """
    renamed = 0
    deleted = 0

    if not dir_path.is_dir():
        return renamed, deleted

    mapping = collect_files(dir_path)

    for base, files in mapping.items():
        # If there is exactly one file and it's already the base name, nothing to do.
        if len(files) == 1 and files[0].name == base:
            continue

        # Separate canonical (no suffix) from suffixed duplicates.
        base_path = dir_path / base
        suffixed: List[Path] = []
        for p in files:
            if p.name == base:
                continue
            # p has some '(1)', '(2)' etc.
            suffixed.append(p)

        # If canonical exists and we also have suffixed: delete suffixed as duplicates.
        if base_path.exists():
            for p in suffixed:
                try:
                    p.unlink()
                    deleted += 1
                    print(f"[DELETE] duplicate {p} (canonical exists: {base_path.name})")
                except Exception as e:
                    print(f"[WARN] failed to delete {p}: {e}")
            continue

        # Canonical does NOT exist. Try to pick one suffixed file to become canonical.
        if suffixed:
            # Pick first suffixed file deterministically (sorted by name)
            suffixed_sorted = sorted(suffixed, key=lambda x: x.name)
            keep = suffixed_sorted[0]
            # Rename keep -> base
            target = base_path
            try:
                keep.rename(target)
                renamed += 1
                print(f"[RENAME] {keep.name} -> {target.name}")
            except Exception as e:
                print(f"[WARN] failed to rename {keep} -> {target}: {e}")

            # Delete any remaining suffixed duplicates
            for p in suffixed_sorted[1:]:
                try:
                    p.unlink()
                    deleted += 1
                    print(f"[DELETE] extra duplicate {p}")
                except Exception as e:
                    print(f"[WARN] failed to delete {p}: {e}")

    return renamed, deleted


def check_pairs(split_root: Path) -> None:
    """
    After fixing names, report any remaining mismatches between
    inputs and outputs for a given split.
    """
    in_dir = split_root / "inputs"
    out_dir = split_root / "outputs"
    if not in_dir.is_dir() or not out_dir.is_dir():
        return

    in_bases = {p.name for p in in_dir.glob("sample_*.npy")}
    out_bases = {p.name for p in out_dir.glob("sample_*.npy")}

    missing_out = sorted(in_bases - out_bases)
    missing_in = sorted(out_bases - in_bases)

    if missing_out:
        print(f"[CHECK] {split_root.name}: {len(missing_out)} inputs without outputs "
              f"(e.g. {missing_out[0]})")
    if missing_in:
        print(f"[CHECK] {split_root.name}: {len(missing_in)} outputs without inputs "
              f"(e.g. {missing_in[0]})")
    if not missing_out and not missing_in:
        print(f"[CHECK] {split_root.name}: all input/output pairs are matched.")


def main() -> None:
    if not ROOT.is_dir():
        print(f"[ERROR] {ROOT} not found. Run this from the repo root where ML_DATASET lives.")
        return

    total_renamed = 0
    total_deleted = 0

    for split in SPLITS:
        split_root = ROOT / split
        if not split_root.is_dir():
            continue
        print(f"\n=== Fixing split: {split} ===")
        for sub in ("inputs", "outputs"):
            dir_path = split_root / sub
            r, d = fix_dir(dir_path)
            total_renamed += r
            total_deleted += d
        check_pairs(split_root)

    print("\n=== Summary ===")
    print(f"Renamed files: {total_renamed}")
    print(f"Deleted duplicate files: {total_deleted}")
    print("Done. Re-run training: "
          "python train_unet_baseline.py --data ML_DATASET --out ML_CHECKPOINTS ...")


if __name__ == "__main__":
    main()

