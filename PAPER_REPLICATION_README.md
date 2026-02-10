# Paper Replication: Khorrami et al. npj Comput. Mater. 2023

This describes how to align **data preparation** and **training** with the paper so you have maximum chance to reach paper accuracy (~1.73 MPa MAE). The **paper replication script** keeps the same strategy as the paper; the **advanced script** applies improvements on top.

---

## 1. Use 10-grain microstructures only

The paper uses **exactly 10 grains** per microstructure (Voronoi). Your data must match.

- **Verify grain count**
  ```bash
  python verify_10grain.py --hdf5-dir simulation_results/hdf5_files
  ```
  If not all seeds have 10 grains, the script reports counts and can write a list of 10-grain seeds with `-o seeds_10grain.txt`.

- **Build dataset with 10-grain filter**
  ```bash
  python build_ml_dataset.py --train 800 --val 200 --max-seeds 1000 --max-grains 10
  ```
  This keeps only seeds with exactly 10 grains and uses 80% train / 20% val (800/200) as in the paper. If you have fewer than 1000 seeds, adjust `--max-seeds`; the split remains 80/20 when you set e.g. `--train 800 --val 200`.

**Seeds vs samples:** The script counts by **seeds** (one seed = one HDF5 = one microstructure). Each HDF5 has many **time increments**; each consecutive pair (t, t+Δt) becomes one training **sample**. So:

- **1000 HDF5** with `--train 800 --val 200` → **800 train seeds**, **200 val seeds**.
- **800 train seeds** × (increments per HDF5 − 1) → **train sample count**.  
  If each simulation has e.g. 20 increments, that’s 800 × 19 ≈ **15,200 train samples**; with 21 increments, 800 × 20 = **16,000 train samples**.

So “800 train” in the paper means **800 microstructures (HDF5 files)**, which yields on the order of **tens of thousands of samples** depending on how many time steps you simulated. Increasing data to ~16,000 samples is exactly what you get when you have 1000 HDF5 and use 800 for training with ~20 increments per run.

---

## 2. Same data preparation as paper

`build_ml_dataset.py` already uses the **same normalization as the paper** (Table 3 ranges, “normalized to 1”):

| Quantity | Range (paper Table 3) | Formula in code |
|----------|------------------------|------------------|
| E        | [50, 300] GPa          | (E − 50e9) / 250e9 |
| ν        | [0.2, 0.4]             | (ν − 0.2) / 0.2   |
| ξ₀       | [50, 300] MPa          | (ξ₀ − 50e6) / 250e6 |
| h₀       | [0, 50] GPa            | h₀ / 50e9         |
| σvM      | (output)               | (σvM in Pa / 1e6) / 1000 → MPa/1000 |

So **no extra data script** is needed for replication: use `build_ml_dataset.py` with `--max-grains 10` and 80/20 split as above.

---

## 3. Same architecture and training in paper replication script

`train_paper_replication.py` is aligned with the paper:

| Paper (Methods) | Replication script |
|------------------|---------------------|
| U-Net, 32 filters | `base=32` |
| 9×9 **separable** 2D convolution | Depthwise 9×9 + pointwise 1×1 (`PaperSepBlock`) |
| Batch norm, ReLU | After each separable block |
| 2D max pooling | `MaxPool2d(2)` |
| Bilinear upsampling | `Upsample(scale_factor=2, mode='bilinear')` |
| Glorot init | `xavier_uniform_` on all Conv/Linear |
| Adam lr=0.001, momentum 0.9 | `Adam(..., lr=0.001, betas=(0.9, 0.999))` |
| Loss: MAE | `L1Loss()` |
| 500 epochs | `--epochs 500` (default) |
| No early stopping | Full 500 epochs |

**Run**
```bash
python train_paper_replication.py --data ML_DATASET --out ML_CHECKPOINTS/paper_replication
```

**MAE in MPa** = validation MAE (normalized) × 1000. Paper reports train 1.733 MPa, val 1.743 MPa.

---

## 3b. Why paper replication can perform *worse* than the baseline (old script)

If you see **better val MAE with the baseline script** (e.g. `train_unet_baseline.py` on 16k data with early stopping) than with the paper replication (500 epochs), it is usually due to:

1. **No early stopping in the paper script**  
   The paper runs a fixed 500 epochs. Your baseline uses **early stopping** (e.g. patience 10), so it saves the best model and stops when validation stops improving. The paper script keeps training → **overfitting**: train MAE keeps dropping (e.g. ~0.005) while val MAE can spike (e.g. 0.08–0.16). So the *reported* “best” from the paper run is still the best *checkpoint* saved (best.pt), but if you compare “last” or late epochs, the baseline wins because it never overfit.

2. **Dataset size and recipe mismatch**  
   The paper is tuned for **800 train / 200 val**, 10-grain-only. If you run the paper script on **the same 16k dataset** you used for the baseline, the architecture and hyperparameters (lr=0.001, 500 epochs) were not designed for that scale. The baseline (e.g. lr=1e-4, early stopping) is better suited to larger data and avoids overfitting.

3. **Architecture vs data**  
   The paper uses **9×9 separable** convolutions; the baseline uses **3×3** standard convolutions. For the paper’s setting (10-grain, 1k samples), 9×9 may be optimal. For more diverse or larger data, 3×3 can generalize better.

**What to do**

- For **paper-like numbers**: use **paper-like data** — 10-grain only, 800/200 split — and run the paper script as-is (or with optional early stopping, see below).
- For **your 16k dataset**: use the **baseline script with early stopping**, or run the paper script **with early stopping** so it doesn’t overfit:  
  `python train_paper_replication.py --data ML_DATASET --out ML_CHECKPOINTS/paper_replication --patience 15`

---

## 4. Advanced script (better than paper)

`train_advanced.py` keeps the **same data** (same `ML_DATASET` and normalization) but uses a different **model and training strategy** (e.g. ResUNet, AdamW, scheduler, warmup, gradient clipping, early stopping) to aim for better than paper accuracy. Use it after you have run the paper replication and want to push below ~1.74 MPa.

---

## 5. Checklist for maximum chance to reach paper accuracy

1. Run `verify_10grain.py` and confirm you have enough 10-grain seeds (e.g. ≥800 train + 200 val).
2. Build dataset with **same normalization** and **10-grain only**:  
   `build_ml_dataset.py --train 800 --val 200 --max-seeds 1000 --max-grains 10`
3. Train with **same strategy** as paper:  
   `train_paper_replication.py --data ML_DATASET --out ML_CHECKPOINTS/paper_replication`  
   (500 epochs, no early stopping.)
4. Compare val MAE × 1000 to 1.743 MPa; then try `train_advanced.py` to go beyond.

---

## 6. Test split and test-set accuracy

When you have a separate test set (e.g. HDF5 in `simulation_results/test_hdf5_files`):

1. **Build only the test split** (does not touch train/val):
   ```bash
   python build_test_split.py --test-hdf5-dir simulation_results/test_hdf5_files --dataset ML_DATASET
   ```

2. **Evaluate on test**  
   - If you have the **paper replication** checkpoint (from `train_paper_replication.py`):
     ```bash
     python evaluate_test.py --data ML_DATASET --ckpt ML_CHECKPOINTS/paper_replication/best.pt --out ML_EVAL --arch paper --split test
     ```
   - If you have the **baseline U-Net** checkpoint (e.g. `ML_CHECKPOINTS/best.pt` from `train_unet_baseline.py`):
     ```bash
     python evaluate_test.py --data ML_DATASET --ckpt ML_CHECKPOINTS/best.pt --out ML_EVAL --arch unet --base 32 --split test
     ```

   Output: test MAE and RMSE (normalized and in MPa) in `ML_EVAL/test_metrics.json` and printed to console.

   **Quick comparison (metrics only, no plots):** add `--metrics-only`.

---

## 7. Comparing two checkpoints (e.g. less data vs more data)

Same test set, different checkpoints. Use the **same** `--arch` and `--base` as each model was trained with.

| Checkpoint | Architecture | Base | Command |
|------------|--------------|------|---------|
| Old / less data | unet | 32 | `python evaluate_test.py --data ML_DATASET --ckpt ML_CHECKPOINTS/best.pt --out ML_EVAL_old --arch unet --base 32 --split test` |
| New / more data | unet | 48 | `python evaluate_test.py --data ML_DATASET --ckpt "ML_CHECKPOINTS/new_data_results/ML_CHECKPOINTS/best.pt" --out ML_EVAL_new --arch unet --base 48 --split test` |

- **Old model** (`ML_CHECKPOINTS/best.pt`, base 32): you already got **Test MAE 24.55 MPa**.
- **New model** (`ML_CHECKPOINTS/new_data_results/ML_CHECKPOINTS/best.pt`, base 48): run the second command above; then compare the printed TEST MAE (MPa) to 24.55. Lower is better.
