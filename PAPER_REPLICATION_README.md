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
