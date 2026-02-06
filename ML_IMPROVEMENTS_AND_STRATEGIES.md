# ML Improvements and Strategies Beyond the Paper

This document outlines **improved and state-of-the-art** techniques (architectures, training, and data strategies) to get **better than paper** results (lower MAE, better generalization) for stress-field surrogate modeling.

---

## 1. Architecture Improvements

### 1.1 Residual U-Net (ResUNet)
- **Idea:** Add residual/skip connections inside each encoder/decoder block (not only U-Net skip across scales). Eases gradient flow and often improves accuracy.
- **Use:** Replace `DoubleConv` with a block that does `out = conv(x) + shortcut(x)` (e.g. 1×1 conv to match channels).
- **Refs:** ResUNet for segmentation; often 5–15% better MAE in regression.

### 1.2 Attention U-Net
- **Idea:** Add channel or spatial attention in the decoder (e.g. at each up stage) so the model focuses on high-error regions (e.g. grain boundaries).
- **Modules:** Squeeze-and-excitation (SE) for channel attention; or spatial attention (e.g. attention gates) using skip features.
- **Benefit:** Paper shows max error at boundaries; attention can weight those regions.

### 1.3 Deeper / Wider U-Net
- **Deeper:** One more down/up stage (e.g. 4 levels instead of 3) for more receptive field.
- **Wider:** Increase base channels (e.g. 32 → 48 or 64) for more capacity. Paper uses 32; our baseline already allows `--base 48/64`.
- **Trade-off:** More parameters and compute; use dropout or weight decay to avoid overfitting.

### 1.4 Fourier Neural Operator (FNO)
- **Idea:** Operate in Fourier space; well-suited to PDEs and periodic microstructures (paper mentions FNO in refs 24–27).
- **Pros:** Can capture long-range dependence; good for homogenization.
- **Cons:** Different codebase; need FNO implementation for 2D image-to-image.

### 1.5 Multi-scale / Pyramids
- **Idea:** Predict at multiple resolutions (e.g. 64×64 and 32×32) with a shared encoder and combine losses (e.g. deep supervision).
- **Benefit:** Encourages consistency across scales and can reduce boundary blur.

---

## 2. Training and Optimization Strategies

### 2.1 Learning Rate Schedule
- **Paper:** Fixed lr=0.001 for 500 epochs.
- **Better:** Cosine annealing or ReduceLROnPlateau (e.g. reduce lr when val MAE stalls for 20–30 epochs). Often gives 10–20% lower final MAE.
- **Warmup:** Short linear warmup (e.g. 5–10 epochs) before constant or cosine decay.

### 2.2 Gradient Clipping
- **Use:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to avoid exploding gradients in deep nets.
- **Typical:** max_norm between 0.5 and 2.0.

### 2.3 Loss Improvements
- **MAE (L1):** Paper and baseline; robust to outliers.
- **Huber:** Smooth L1; differentiable at 0; can help stability.
- **Boundary-weighted loss:** Weight pixel loss by distance-to-boundary or a boundary mask so errors at grain boundaries are penalized more (targets paper’s main error region).
- **Combined:** e.g. `L = L_MAE + 0.1 * L_boundary` or multi-scale losses.

### 2.4 Regularization
- **Weight decay:** Paper does not state; 1e-5 to 1e-4 often helps.
- **Dropout:** Small dropout (0.1–0.2) in decoder or bottleneck can improve generalization.
- **Data augmentation:** See §3.

### 2.5 Batch Size and Normalization
- **Larger batch:** 16 or 32 if memory allows; more stable batch norm and gradients.
- **BatchNorm:** Paper uses it; keep it. Alternative: GroupNorm if batch size is very small (e.g. 2–4).

### 2.6 Mixed Precision (AMP)
- **Use:** `torch.amp.autocast('cuda')` and GradScaler for faster training and lower memory; usually same or better MAE.

---

## 3. Data and Augmentation

### 3.1 Geometric Augmentation
- **Rotations:** 0°, 90°, 180°, 270° (consistent for input and target).
- **Flips:** Horizontal/vertical (microstructure can be symmetric).
- **Benefit:** Effectively multiplies dataset size; improves invariance and often 5–15% MAE gain.

### 3.2 Normalization and Scaling
- **Input normalization:** Per-channel mean/std (or min–max) on training set; apply same to val/test. Paper “normalized to 1”; ensure val/test use training statistics.
- **Stress range:** Paper notes very few pixels >500 MPa; consider log-scale or robust scaling for σvM to balance gradients.

### 3.3 Temporal / Increment Sampling
- **Idea:** Sample (t, t+Δt) pairs with variable Δt (e.g. 1, 2, 3 steps) so the model sees multiple time scales. Can improve extrapolation to larger F11.

### 3.4 Curriculum or Hard Example Mining
- **Curriculum:** Train first on “easier” increments (e.g. early strain), then add harder ones.
- **Mining:** Oversample or reweight samples with high error (e.g. boundary-heavy crops).

---

## 4. Evaluation and Validation

### 4.1 Metrics Beyond MAE
- **RMSE:** Penalizes large errors more.
- **Relative error:** e.g. mean(|pred − gt| / (|gt| + ε)) per pixel or per sample.
- **Percentiles:** 90th/95th percentile error (paper reports max error at boundaries).
- **Per-seed or per-microstructure:** Report MAE distribution across seeds to detect hard morphologies.

### 4.2 Stress–Strain Consistency
- **Macroscopic σvM vs strain:** Compare mean stress over the cell (or over grains) for GT vs predicted; check elastic slope and hardening (paper Fig. 8 style).
- **Use:** Validates that the surrogate is not only accurate per pixel but also mechanically consistent.

### 4.3 Speed Benchmark
- **Paper:** ~500× speedup (tCNN vs DAMASK). Report inference time per sample and per DAMASK run on your hardware for fair comparison.

---

## 5. Advanced Script: What `train_advanced.py` Implements

The script `train_advanced.py` includes:

- **Residual U-Net:** Residual blocks in encoder/decoder for better gradient flow and accuracy.
- **Larger capacity:** Base channels 48 (configurable) for more capacity than the paper’s 32.
- **Learning rate schedule:** Cosine annealing or ReduceLROnPlateau after optional warmup.
- **Gradient clipping:** Configurable max norm.
- **Optional boundary-weighted loss:** Mask or weight by distance to boundary (if you provide or compute it).
- **AdamW:** Weight decay decoupled (often better than Adam + L2).
- **More epochs + early stopping:** e.g. 500 epochs with patience 30–50 to avoid underfitting while stopping when val MAE plateaus.
- **Checkpointing and logging:** Best and last checkpoint; CSV log; optional TensorBoard.

These choices aim to **beat the paper’s reported val MAE (1.743 MPa)** when trained on the same or similar data (e.g. 800 train / 100 val seeds, 10-grain-like or your current microstructure set).

---

## 6. Quick Reference: Paper vs Advanced

| Aspect | Paper (Khorrami et al.) | Advanced script |
|--------|-------------------------|------------------|
| Architecture | U-Net, 32 filters, 9×9 conv | ResUNet-style, 48 base, 3×3 (+ optional 9×9 first) |
| Init | Glorot | He/Xavier or Glorot |
| Optimizer | Adam lr=0.001 | AdamW + cosine or plateau LR |
| Epochs | 500 fixed | 500 + early stopping |
| Loss | MAE | MAE (+ optional boundary weight) |
| Regularization | — | Weight decay, grad clip, optional dropout |
| Goal | Replicate 1.73 MPa | Target &lt;1.5 MPa (better than paper) |

---

## 7. References (from paper and related)

- Khorrami et al., npj Comput. Mater. (2023) – baseline paper.
- Mianroodi et al., npj Comput. Mater. (2021) – U-Net for stress.
- Kapoor et al., arXiv:2210.16994 – U-Net vs FNO for stress.
- Ronneberger et al., U-Net (2015) – original architecture.
- ResUNet, Attention U-Net – standard improvements in medical/segmentation that transfer to regression.
