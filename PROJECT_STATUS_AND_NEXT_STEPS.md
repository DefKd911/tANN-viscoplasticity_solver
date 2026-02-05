# Project Status & Next Steps (ML Modelling)

**Date:** Feb 3, 2025

---

## 1. Directory & asset status

| Asset | Location | Count | Status |
|-------|----------|--------|--------|
| **Microstructures (VTI)** | `geom_vti/` | 1,000 | ✅ Present |
| **Material YAML** | `material_yaml/` and `material_yaml_fixed/` | 1,000 each | ✅ Present |
| **Grain labels** | `labels/` | 1,000 (.npy) | ✅ Present |
| **Grain properties** | `props/` | 1,000 (.npy) | ✅ Present |
| **DAMASK HDF5 results** | `simulation_results/hdf5_files/` | 1,000 | ✅ All runs done |
| **Simulation logs** | `simulation_results/simulation_logs/` | (per run) | ✅ |
| **Stress curves** | `simulation_results/stress_curves/` | 46 CSV+PNG | ✅ |
| **ML dataset** | `ML_DATASET/` | train/val/test + metadata | ✅ Built from 50 seeds |
| **ML checkpoints** | `ML_CHECKPOINTS/` | best.pt, last.pt, training_log.csv | ✅ Trained 50 epochs |
| **ML evaluation** | `ML_EVAL/` | predictions, panels, videos, metrics | ✅ Evaluated |

**Current split:** Use **900 seeds** → **800 train**, **100 val**, **test prepared later**. Run `build_ml_dataset.py` (defaults: `--train 800 --val 100 --test 0 --max-seeds 900`).

---

## 2. Current ML pipeline (what’s done)

1. **Data:** `build_ml_dataset.py` reads from `simulation_results/hdf5_files/`, `labels/`, `props/` → builds `ML_DATASET` (inputs 64×64×5, outputs 64×64×1).
2. **Train:** `train_unet_baseline.py` — U-Net, L1 loss, Adam, early stopping. Trained 50 epochs; best val MAE ≈ 0.0159 (epoch 48).
3. **Eval:** `evaluate_test.py` — test MAE **14.9 MPa**, RMSE **21.5 MPa**; per-seed and per-sample analysis, visualizations, stress–strain curves, videos.

So: **simulation → HDF5 → dataset build → training → evaluation** is implemented and run.

---

## 3. Recommended next steps (in order)

### A. More data (highest impact)

- **Run more DAMASK simulations** so the model sees more microstructures.
  - You have 1,000 microstructures; only 50 are in HDF5. Use `batch_run_damask.py` (or your existing batch setup) to run more seeds.
  - Target: e.g. 100–200+ HDF5 runs, then re-run `build_ml_dataset.py` and retrain.
- **Rebuild dataset:** After new HDF5 files are in `simulation_results/hdf5_files/`, run:
  - `python build_ml_dataset.py`
- **Retrain:** Use the same or updated script:
  - `python train_unet_baseline.py --data ML_DATASET --epochs 80 --batch-size 8`

### B. Model improvements (architecture & training)

- **Larger U-Net:** In `train_unet_baseline.py`, try `--base 48` or `--base 64` (wider channels).
- **Residual blocks:** Replace or wrap `DoubleConv` with residual blocks to help gradient flow and stability.
- **Learning rate:** Try a small scheduler (e.g. ReduceLROnPlateau on val MAE) and/or slightly lower initial LR (e.g. 5e-5).
- **Longer training:** Increase `--epochs` (e.g. 80–100) and optionally increase early-stopping patience.

### C. Inputs & features (better use of microstructure)

- **Grain-boundary / distance channels:** Add a channel (e.g. distance to grain boundary or boundary mask) and feed 6-channel input; update first conv in the U-Net from 5 to 6.
- **Temporal history:** Use 2–3 past stress steps as extra channels (e.g. σ(t), σ(t−1), σ(t−2)) so input is 7–8 channels; update model and dataset builder accordingly.
- **Normalization:** Already in place in `metadata/normalization.json`; keep reusing it when adding new channels.

### D. Boundary-focused loss (target known weakness)

- **Boundary weighting:** In the loss, weight pixels by distance-to-boundary or a boundary mask so errors at boundaries are penalized more (e.g. weighted L1).
- **Auxiliary boundary head:** Add a small head that predicts a boundary map and train with an extra loss term; can help the main stress head focus on boundaries.

### E. Evaluation & iteration

- **Stratified analysis:** Keep evaluating by seed and by “easy” vs “hard” microstructure (e.g. equiaxed vs elongated) to see where gains come from.
- **Stress–strain consistency:** Keep comparing macroscopic stress–strain (GT vs predicted mean stress) after each change.
- **Videos:** Keep generating a few stress evolution videos after major changes to check temporal coherence and boundary behavior.

### F. Automation (optional)

- **Single script/notebook:** One entry point that (1) builds dataset from HDF5, (2) trains, (3) runs evaluation and saves metrics/figures.
- **Config file:** Move hyperparameters (base channels, LR, epochs, paths) into a YAML/JSON so you can sweep or version experiments easily.

---

## 4. Quick reference commands

### Dumping more HDF5 simulation files

From the project root (where `geom_vti/`, `material_yaml_fixed/`, `load.yaml`, `numerics.yaml` are):

```bash
# Run more simulations (non-interactive). Skips seeds that already have HDF5.
python batch_run_damask.py --max 200

# Run all remaining geometries that don't have HDF5 yet
python batch_run_damask.py --all

# Faster: run up to 100 new simulations in parallel (4 workers)
python batch_run_damask.py --max 100 --parallel
```

Interactive (prompts for count and mode):

```bash
python batch_run_damask.py
```

HDF5 outputs go to `simulation_results/hdf5_files/` (e.g. `seed123456.hdf5`). Already-existing files are skipped.

### ML dataset and training

```bash
# Build ML dataset: 900 seeds -> 800 train, 100 val, 0 test (default)
python build_ml_dataset.py

# Custom split, e.g. 700 train, 150 val, 50 test, use first 900 seeds
python build_ml_dataset.py --train 700 --val 150 --test 50 --max-seeds 900
# Use all HDF5 seeds (e.g. 1000): --max-seeds 0

# Train (default 50 epochs, base 32)
python train_unet_baseline.py --data ML_DATASET --epochs 50 --batch-size 8

# Train larger model, more epochs
python train_unet_baseline.py --data ML_DATASET --epochs 80 --batch-size 8 --base 48

# Evaluate on test set
python evaluate_test.py
```

---

## 5. Summary

- **Status:** Simulation pipeline, HDF5 dumps, microstructures (YAML + VTI), labels, and ML dataset/training/eval are in place. Baseline U-Net gives ~14.9 MPa test MAE.
- **Bottleneck:** Only 50 of 1,000 microstructures have been simulated; expanding to 100–200+ is the most impactful next step.
- **Next focus:** (1) Run more DAMASK jobs and rebuild dataset, (2) Retrain with same or larger U-Net and optional LR schedule, (3) Add boundary-aware or temporal features and/or boundary-aware loss, then (4) Re-evaluate and iterate.
