# Paper Replication Training Report

This document summarizes the **paper replication** training run: where the best validation error occurred, whether training was correct, what is saved, and whether further or improved training is needed.

---

## 1. Best validation result (lowest error)

| Metric | Value |
|--------|--------|
| **Best epoch** | **292** |
| **Best validation MAE (normalized)** | **0.00346172** |
| **Best validation MAE (MPa)** | **3.46 MPa** (normalized × 1000) |

The **lowest validation MAE** in your run occurred at **epoch 292**.  
Paper reference: validation MAE ≈ **1.743 MPa**. Your best is ~3.46 MPa on your dataset (different data and conditions explain the gap).

---

## 2. Training progress and history (how you reached low error)

- **Epochs 1–30:** Fast drop. Val MAE goes from ~0.021 to ~0.008–0.012. Train MAE ~0.014–0.056.
- **Epochs 30–100:** Steady improvement. Val MAE fluctuates between ~0.007 and ~0.020; several epochs near or below 0.008 (e.g. 69: 0.00655, 101: 0.00739).
- **Epochs 100–200:** Further improvement. Val MAE often in 0.007–0.010; train MAE keeps decreasing (~0.008–0.010).
- **Epochs 200–292:** Best region. Val MAE reaches **0.00346 at epoch 292** (best overall). Other good epochs in this range: 253 (0.00579), 283 (0.00574), 297 (0.00413), 301 (0.00443).
- **Epochs 292–500:** **Overfitting**. Train MAE keeps decreasing (down to ~0.005); val MAE becomes noisy and often much higher (e.g. 0.04–0.16). So the *best* model is from epoch 292, not from the end of training.

So the low validation error was reached by **normal training** (no bug): the model improved until around epoch 292, then kept fitting the training set and validation got worse.

---

## 3. Is this correct training?

**Yes.** The run is correct:

- Train and val MAE both improve in the first half of training.
- The best checkpoint is saved when val MAE improves (epoch 292).
- After that, train MAE keeps going down while val MAE degrades or spikes — that is **overfitting**, which is expected when you run 500 epochs with no early stopping.

So nothing is “wrong”; the script did what it is designed to do. The fact that validation gets worse after epoch 292 is why we recommend using **best.pt** (from epoch 292) and optionally **early stopping** (e.g. `--patience 15`) for future runs.

---

## 4. Where the best model is saved

Everything is under:

**`ML_CHECKPOINTS/paper_replication/`**

| File | Meaning |
|------|--------|
| **`best.pt`** | Model and metadata from the **epoch with lowest validation MAE** (epoch 292). This is the one to use for evaluation and comparison. Contains: `model` (state_dict), `epoch` (292), `val_mae` (~0.00346). |
| **`last.pt`** | Model at the **end** of training (epoch 500). Validation is worse here due to overfitting; do **not** use this for reporting best performance. |
| **`training_log.csv`** | Every epoch: `epoch`, `train_mae`, `val_mae`. Use this to plot curves or find the best epoch. |
| **`training_curve.png`** | Plot of train and val MAE vs epoch (if the script wrote it). |

For **evaluation and comparison**, always use **`best.pt`**, not `last.pt`.

---

## 5. Do you need better or improved training? Will the previous run still work?

- **Previous run is still valid.**  
  The run you did is correct. The model to use is **`ML_CHECKPOINTS/paper_replication/best.pt`** (epoch 292, val MAE ≈ 3.46 MPa). You do not need to discard this run.

- **Optional improvements for *future* runs:**
  - **Early stopping:**  
    `python train_paper_replication.py --data ML_DATASET --out ML_CHECKPOINTS/paper_replication_early --patience 15`  
    Training will stop when val MAE does not improve for 15 epochs. You avoid the long overfitting tail (292→500) and may get similar or slightly better best val MAE with less compute.
  - **Advanced script:**  
    `train_advanced.py` uses a different architecture and training strategy (e.g. ResUNet, AdamW, scheduler, early stopping) and may achieve lower MAE on your data. Use it **in addition to** (not instead of) the paper replication if you want to push performance further.

- **What to do with the current best model:**
  - Run **test-set evaluation** with `best.pt` to get **test MAE in MPa** on a held-out test set.
  - Compare that to your baseline (e.g. `ML_CHECKPOINTS/new_data_results/ML_CHECKPOINTS/best.pt`) on the **same test set** to see which model is better in practice.

---

## 6. What to do next: evaluate first, then decide on further training

### Paper reference (Khorrami et al. npj Comput. Mater. 2023)

The paper reports **training** MAE **1.733 MPa** and **validation** MAE **1.743 MPa**. Your current best is **validation ~3.46 MPa** (epoch 292). The gap is expected when data and conditions differ; the important thing is to have a clear, comparable evaluation before changing anything.

### Recommended order

1. **First: evaluate accuracy on validation and test (no new training)**  
   Run the evaluator on your current **best.pt** so you have one number for validation and one for test. That way you know exactly how this model performs.

   **On validation split:**
   ```bash
   python evaluate_test.py --data ML_DATASET --ckpt ML_CHECKPOINTS/paper_replication/best.pt --out ML_EVAL_PAPER --arch paper --split val
   ```
   This prints **VAL MAE (MPa)** and writes `ML_EVAL_PAPER/val_metrics.json` and `val_metrics.csv`.

   **On test split** (if you have a test set under `ML_DATASET/test`):
   ```bash
   python evaluate_test.py --data ML_DATASET --ckpt ML_CHECKPOINTS/paper_replication/best.pt --out ML_EVAL_PAPER --arch paper --split test
   ```
   This prints **TEST MAE (MPa)** and writes `ML_EVAL_PAPER/test_metrics.json` and `test_metrics.csv`.

   If you don’t have a test set yet, run only the validation command above. The validation MAE from this evaluation is the proper “accuracy on validation” for your current best model.

2. **Then: decide on further training**
   - **If the val/test MAE is acceptable for your use** → You can stop. Use **best.pt** for inference; no need for improvised training now.
   - **If you want to get closer to the paper (1.74 MPa) or improve further** → Then consider:
     - **Option A:** Run paper replication again with early stopping:  
       `python train_paper_replication.py --data ML_DATASET --out ML_CHECKPOINTS/paper_replication_early --patience 15`
     - **Option B:** Run the advanced script:  
       `python train_advanced.py` (see PAPER_REPLICATION_README.md).  
   After any new training, run the same evaluation commands (val and test) on the new checkpoint and compare to the numbers you got in step 1.

### Summary

| Step | Action |
|------|--------|
| 1 | **Evaluate first:** run `evaluate_test.py` with `best.pt` on **val** and **test** (if available). Record val MAE (MPa) and test MAE (MPa). |
| 2 | **Then decide:** only if you need better numbers, run improved training (early stopping or `train_advanced.py`) and evaluate the new checkpoint the same way. |

So: **do not** go for further training with improvised techniques until you have run evaluation on validation (and test) and have those numbers. After that, you can decide whether more training is worth it.

---

## 7. Summary table

| Item | Value |
|------|--------|
| Best epoch | 292 |
| Best val MAE (normalized) | 0.00346172 |
| Best val MAE (MPa) | 3.46 MPa |
| Paper val MAE (reference) | 1.743 MPa |
| Checkpoint to use | `ML_CHECKPOINTS/paper_replication/best.pt` |
| Training correct? | Yes |
| Overfitting after epoch 292? | Yes (expected without early stopping) |
| Next step | **Evaluate on val and test first** (see §6); then decide on further training |
| Previous run still valid? | Yes; use `best.pt` for all evaluation and comparison |

---

*Generated from `ML_CHECKPOINTS/paper_replication/training_log.csv` and the paper replication script behavior.*
