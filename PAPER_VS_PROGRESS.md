# Paper vs Current Progress: Khorrami et al. (npj Comput. Mater. 2023)

**Paper:** “An artificial neural network for surrogate modeling of stress fields in viscoplastic polycrystalline materials”  
**Your project:** tANN viscoplasticity solver – U-Net surrogate for von Mises stress from DAMASK.

---

## 1. What the paper does (summary)

| Aspect | Paper |
|--------|--------|
| **Physics** | J2 elasto-viscoplasticity, linear isotropic hardening; quasi-static uniaxial extension |
| **Solver** | DAMASK, spectral (FFT) method = “reference model” (RM) |
| **Inputs** | E, ν, ξ0, h0 (per grain → pixel maps), σvM(t) |
| **Output** | σvM(t+Δt) |
| **Resolution** | 64×64 |
| **Grains (training)** | 10-grain microstructures |
| **Data** | 1000 microstructures → 80% train, 20% validation |
| **Extension rate** | Single rate 1×10⁻³ s⁻¹; training up to F11 = 1.004 |
| **Material ranges** | E ∈ [50, 300] GPa, ν ∈ [0.2, 0.4], ξ0 ∈ [50, 300] MPa, h0 ∈ [0, 50] GPa |
| **Architecture** | U-Net, 32 filters, 9×9 separable conv, batch norm, bilinear upsampling (TensorFlow) |
| **Training** | Adam lr=0.001, MAE loss, 500 epochs |
| **Train MAE** | **1.733 MPa** |
| **Val MAE** | **1.743 MPa** |
| **Speed** | tCNN ~**500×** faster than RM (0.15 s vs 75 s per run) |

**Extra in paper (you have not done yet):**

- Test sets: **5- and 20-grain** microstructures (50 each); MAE ~1.58 MPa (5 grain), ~1.87 MPa (20 grain).
- **Material contrast** test cases (4 cases, subsets of property ranges); MAE from ~1.27 to ~3.66 MPa.
- **Morphology** tests: matrix–inclusion (circular and square); MAE ~1.42 MPa and ~1.58 MPa.
- **Extension beyond training** (F11 > 1.004); error growth and stress–strain curve analysis.
- **Computational efficiency**: explicit timing vs DAMASK.

---

## 2. What you have done (current progress)

| Aspect | Your project |
|--------|----------------|
| **Physics** | ✅ Same: J2 viscoplasticity, DAMASK, quasi-static uniaxial extension |
| **I/O** | ✅ Same: (E, ν, ξ0, h0, σvM(t)) → σvM(t+Δt), 64×64 |
| **Material ranges** | ✅ Same (Table 3) |
| **Data** | ✅ 1000 microstructures; **900 simulated** (800 train + 100 val), **100 test** (separate folder, for evaluation) |
| **Train/val split** | ✅ 800 seeds (~16k samples) train, 100 seeds (~2k) val |
| **Test set** | ✅ 100 seeds defined, separate HDF5 folder; not yet fully run + evaluated |
| **Architecture** | ✅ U-Net (PyTorch), base 32, L1 (MAE) loss, Adam |
| **Training** | ✅ Trained; checkpoints in `ML_CHECKPOINTS/new_data_results/` |
| **Val MAE** | **~4.85 MPa** (normalized 0.00485 → ×1000 MPa) |
| **Speed** | ❌ Not measured vs DAMASK |

---

## 3. Gap: accuracy (MAE)

| Metric | Paper | You | Gap |
|--------|--------|-----|-----|
| Train MAE | 1.733 MPa | (not reported in same form) | — |
| Val MAE | 1.743 MPa | **~4.85 MPa** | **~2.8× higher** |
| Test MAE (10-grain–like) | 1.733 MPa | Pending (100 test seeds) | — |

So relative to the paper, you are **roughly 2.8× worse on validation MAE** (4.85 vs 1.74 MPa). Possible reasons (without changing data):

- Different U-Net details (e.g. 3×3 conv vs 9×9 separable, depth, width).
- Different training (epochs, lr, schedule) – paper 500 epochs, Adam 0.001.
- Different normalization or data distribution (increment spacing, strain range).
- Paper: fixed 10 grains; yours: variable grain count (Voronoi-like), so more grain-boundary variety.

---

## 4. Checklist: paper vs you

| Paper element | Your status |
|---------------|-------------|
| DAMASK + J2 viscoplasticity, 64×64 | ✅ Done |
| E, ν, ξ0, h0, σvM(t) → σvM(t+Δt) | ✅ Done |
| Train/val from same physics & ranges | ✅ Done |
| U-Net, MAE loss, Adam | ✅ Done |
| Reported train/val MAE (~1.73 MPa) | ⚠️ You ~4.85 MPa val (worse) |
| Test on **5- and 20-grain** microstructures | ❌ Not done |
| Test on **material contrast** subsets | ❌ Not done |
| Test on **matrix–inclusion** morphologies | ❌ Not done |
| Test **extension beyond training** (F11 > 1.004) | ❌ Not done |
| **Speed comparison** (tCNN vs DAMASK) | ❌ Not done |
| 500 epochs / lr 0.001 (paper setting) | ⚠️ You used different epochs/lr |

---

## 5. Recommended next steps (to align with paper)

1. **Test set evaluation**  
   Run the 100 test seeds (if not already), build test split from `simulation_results/test_hdf5_files/`, run `evaluate_test.py` → report **test MAE (MPa)** and compare to paper’s 1.73 MPa.

2. **Improve val/test MAE toward paper**  
   - Try paper-like hyperparameters: more epochs (e.g. 500), Adam lr=0.001.  
   - Try larger/wider U-Net (e.g. 9×9 or more channels) if you can match paper’s architecture more closely.  
   - Keep same normalization and data pipeline for a fair comparison.

3. **Match paper’s extra experiments (optional but good for “how far we are”)**  
   - **Grain count:** Generate (or select) 5- and 20-grain microstructures; run DAMASK; evaluate tCNN MAE (paper: Table 1).  
   - **Material contrast:** Define 4 property subsets (paper Eq. 1), generate test data, report MAE (paper: Table 2).  
   - **Morphology:** Single inclusion (circular/square) in matrix; generate + test (paper: Figs. 6–7, MAE ~1.42 / 1.58 MPa).  
   - **Extension beyond training:** Run DAMASK to F11 > 1.004 (e.g. 1.006–1.008); compare σvM and stress–strain with tCNN (paper: Fig. 8).

4. **Speed benchmark**  
   Time one DAMASK run (one microstructure, one load step or full path) and one tCNN forward pass (same resolution) on the same machine → report ratio (paper: ~500×).

5. **Short “Results” section for your report**  
   - Table: Train / Val / Test MAE (MPa) – you vs paper.  
   - One sentence on speed (e.g. “tCNN is ~X× faster than DAMASK on our setup”).  
   - Optional: one figure each for 5/20-grain and material-contrast MAE if you add those tests.

---

## 6. One-line summary

**You have implemented the same problem and pipeline as the paper (DAMASK → U-Net, same I/O and ranges) and have a working trained model, but validation MAE is ~2.8× higher than the paper (~4.85 vs ~1.74 MPa). Test MAE and all extra paper experiments (grain count, contrast, morphology, extension, speed) are still to be done.**
