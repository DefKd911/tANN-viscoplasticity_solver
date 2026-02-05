# Improving Training Speed & GPU Utilization (RTX 4060 8GB)

This doc summarizes ways to make `train_unet_baseline.py` faster and better use your **RTX 4060 (8GB VRAM)**.

---

## 1. Quick command (recommended for RTX 4060)

Use a larger batch size, data-loader workers, and mixed precision so the GPU stays busy and each step is cheaper:

```bash
python train_unet_baseline.py --data ML_DATASET --out ML_CHECKPOINTS --epochs 80 --batch-size 16 --lr 1e-4 --base 48 --patience 10 --num-workers 4 --amp
```

- **`--batch-size 16`** — Fills more of the 8GB VRAM; often 2× faster per epoch than 8.
- **`--amp`** — Automatic mixed precision (FP16); faster and uses less VRAM so you can raise batch size.
- **`--num-workers 4`** — Loads next batches on CPU while GPU trains; reduces “GPU waiting for data”.
- **Pin memory** is enabled automatically when using CUDA (faster CPU→GPU transfer).

If you hit **out-of-memory (OOM)** with 16, try `--batch-size 12` or keep 8.

---

## 2. Options added to the training script

| Option | Default | Recommendation (RTX 4060 8GB) |
|--------|---------|-------------------------------|
| `--batch-size` | 8 | **12–16** (or 24 with `--amp`) to use more VRAM and speed up. |
| `--num-workers` | 0 | **2–4** so data is ready while GPU is busy. |
| `--pin-memory` | off (auto on for CUDA) | On by default when `device=cuda`; no need to set. |
| `--amp` | off | **Use** for faster training and lower VRAM (FP16). |

**Example: maximum throughput (if VRAM allows):**

```bash
python train_unet_baseline.py --data ML_DATASET --out ML_CHECKPOINTS --epochs 80 --batch-size 24 --lr 1e-4 --base 48 --num-workers 4 --amp
```

---

## 3. Batch size vs VRAM (8GB)

Approximate usage for input size 64×64×5, base 48:

| Batch size | Est. VRAM | Notes |
|------------|-----------|--------|
| 8 | ~2–3 GB | Safe default. |
| 16 | ~4–5 GB | Good balance for RTX 4060. |
| 24 | ~6–7 GB | Fits with `--amp`. |
| 32 | ~8+ GB | May OOM; try only with `--amp`. |

If you see **CUDA out of memory**:

- Lower `--batch-size` (e.g. 16 → 12 → 8).
- Use `--amp` if you aren’t already.
- Slightly reduce model size, e.g. `--base 32` instead of 48.

---

## 4. DataLoader: `num_workers` and `pin_memory`

- **`num_workers=0`** (default): single-threaded loading; GPU often waits for the next batch.
- **`num_workers=2` or `4`**: next batches are prepared in parallel; GPU stays busy. On Windows, 2–4 is usually enough; avoid very high values.
- **`pin_memory=True`**: uses pinned (page-locked) CPU memory so transfer to GPU is faster. The script turns this on automatically when using CUDA.

---

## 5. Mixed precision (`--amp`)

- Uses **FP16** for most ops and **FP32** where needed (e.g. loss scaling).
- **Benefits**: higher throughput and **lower VRAM**, so you can use a larger batch size.
- **Quality**: same final MAE in practice for this U-Net; no need to change learning rate for typical use.

---

## 6. Learning rate when changing batch size

If you **increase batch size** (e.g. 8 → 16), you can optionally scale the learning rate (linear scaling rule):

- Double batch → try `--lr 2e-4` (or keep `1e-4` and only change if val loss is unstable).

Default `--lr 1e-4` is fine for `--batch-size 16`; only adjust if you see bad convergence or instability.

---

## 7. Summary: recommended commands

**Conservative (stable, no OOM risk):**
```bash
python train_unet_baseline.py --data ML_DATASET --out ML_CHECKPOINTS --epochs 80 --batch-size 8 --base 48 --num-workers 2 --amp
```

**Faster (RTX 4060 8GB):**
```bash
python train_unet_baseline.py --data ML_DATASET --out ML_CHECKPOINTS --epochs 80 --batch-size 16 --base 48 --num-workers 4 --amp
```

**Maximum throughput (if no OOM):**
```bash
python train_unet_baseline.py --data ML_DATASET --out ML_CHECKPOINTS --epochs 80 --batch-size 24 --base 48 --num-workers 4 --amp
```

---

## 8. Checking GPU utilization

While training, open **Task Manager** → Performance → GPU, or use:

```bash
nvidia-smi -l 2
```

- **GPU utilization** should be high (e.g. 90%+) most of the time. If it dips a lot, try increasing `--num-workers` or `--batch-size`.
- **VRAM** should be well used but below 8GB; if you hit OOM, lower `--batch-size` or use `--amp`.

---

## 9. Other tips

- **Persistent workers**: The script sets `persistent_workers=True` when `num_workers > 0` so worker processes are reused and startup cost is paid once.
- **Same checkpoint format**: All options (amp, batch size, workers) produce the same `best.pt` / `last.pt`; evaluation with `evaluate_test.py` is unchanged.
- **Reproducibility**: For fixed seeds you’d need to set `torch.manual_seed` and numpy seed; the script does not set them by default.
