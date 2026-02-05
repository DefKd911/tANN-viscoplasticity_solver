# Per-Grain Stress-Strain Curve Analysis for Test Samples

## Overview

This script analyzes per-grain stress-strain curves for test microstructures, comparing **predicted** (ML model) vs **actual** (DAMASK ground truth) stress-strain behavior for each individual grain.

## Purpose

- **Validate ML model performance at grain level**: Understand how well the model predicts individual grain behavior
- **Identify grain-specific errors**: Find which grains are predicted well vs poorly
- **Compare predicted vs actual curves**: Visual comparison of stress-strain evolution per grain
- **Generate detailed analysis**: Per-grain error metrics (MAE, RMSE)

## Usage

### Basic Usage

```bash
python analyze_test_per_grain_ss_curves.py
```

This will analyze all test seeds automatically.

### Custom Usage

```bash
python analyze_test_per_grain_ss_curves.py \
    --data ML_DATASET \
    --predictions ML_EVAL/predictions \
    --hdf5-dir simulation_results/hdf5_files \
    --labels-dir labels \
    --props-dir props \
    --out ML_EVAL/per_grain_test_analysis \
    --seeds 1099931136,1098426137,1098322009,1098694165,1102020331 \
    --max-grains 20
```

### Arguments

- `--data`: ML dataset root directory (default: `ML_DATASET`)
- `--predictions`: Directory containing ML predictions (default: `ML_EVAL/predictions`)
- `--hdf5-dir`: Directory with DAMASK HDF5 files (default: `simulation_results/hdf5_files`)
- `--labels-dir`: Directory with grain label files (default: `labels`)
- `--props-dir`: Directory with grain property files (default: `props`)
- `--out`: Output directory for results (default: `ML_EVAL/per_grain_test_analysis`)
- `--metadata-csv`: Path to metadata CSV (default: `ML_DATASET/metadata/increments_map.csv`)
- `--seeds`: Comma-separated list of seeds to analyze (default: all test seeds)
- `--max-grains`: Maximum number of grains to plot per seed (default: 20)

## Output Structure

```
ML_EVAL/per_grain_test_analysis/
├── seed1099931136/
│   ├── grain_000_comparison.png
│   ├── grain_001_comparison.png
│   ├── ...
│   └── summary.json
├── seed1098426137/
│   ├── grain_000_comparison.png
│   ├── ...
│   └── summary.json
├── ...
└── overall_summary.json
```

### Output Files

1. **Per-Grain Comparison Plots** (`grain_XXX_comparison.png`):
   - Ground truth stress-strain curve (blue, solid line with markers)
   - Predicted stress-strain curve (purple, dashed line with markers)
   - Elastic estimate line (gray, dashed)
   - Yield stress line (orange, dotted)
   - Grain properties in title (E, ν, ξ₀, h₀)

2. **Per-Seed Summary** (`summary.json`):
   ```json
   {
     "0": {
       "mae": 12.5,
       "rmse": 15.3,
       "n_points_gt": 25,
       "n_points_pred": 20
     },
     ...
   }
   ```

3. **Overall Summary** (`overall_summary.json`):
   - Aggregated results across all seeds
   - Per-seed summaries

## How It Works

### 1. Ground Truth Curves (from HDF5)

- Loads all increments from DAMASK HDF5 file
- Extracts von Mises stress and strain (ε₁₁) for each voxel
- Averages stress and strain per grain across all increments
- Creates stress-strain curve: `σ_vM` vs `ε`

### 2. Predicted Curves (from ML Predictions)

- Loads all test samples for a given seed
- Extracts predicted stress from saved predictions (`.npy` files)
- Reconstructs stress-strain curve by:
  - Loading initial stress from input data (increment 0)
  - Loading predictions for each subsequent increment
  - Using actual strain values from HDF5 (if available) or estimating from increment index
- Averages predicted stress per grain

### 3. Comparison & Metrics

- Interpolates both curves to common strain points
- Computes MAE and RMSE per grain
- Generates comparison plots

## Example Output

Each plot shows:
- **X-axis**: Strain (%)
- **Y-axis**: Von Mises Stress (MPa)
- **Blue line (GT)**: Ground truth from DAMASK
- **Purple line (Pred)**: ML model prediction
- **Gray line**: Elastic estimate (E × ε)
- **Orange line**: Yield stress (ξ₀)

## Interpretation

### Good Prediction
- Predicted curve closely follows ground truth
- Low MAE/RMSE (< 15 MPa)
- Correct elastic slope
- Accurate yield point prediction

### Poor Prediction
- Large deviation between predicted and actual
- High MAE/RMSE (> 25 MPa)
- Incorrect elastic or plastic behavior
- May indicate boundary effects or complex grain interactions

## Requirements

- Python 3.7+
- numpy
- h5py
- matplotlib
- scipy (optional, for better interpolation)

## Notes

- The script processes up to `--max-grains` grains per seed (default: 20) to avoid too many plots
- Strain values are extracted from HDF5 when available, otherwise estimated from increment indices
- Predictions are denormalized using the same normalization scheme as the dataset
- Grain properties (E, ν, ξ₀, h₀) are displayed in plot titles for reference

## Troubleshooting

### "Labels not found"
- Check that label files exist in `--labels-dir`
- File naming: `labels_seed{seed}.npy` or `seed{seed}.npy`

### "Props not found"
- Check that property files exist in `--props-dir`
- File naming: `props_seed{seed}.npy` or `seed{seed}.npy`

### "HDF5 not found"
- Verify HDF5 files exist in `--hdf5-dir`
- File naming: `seed{seed}.hdf5`

### "No test samples found"
- Check that test samples exist in metadata CSV
- Verify seed names match between metadata and files

### Empty curves
- Some grains may have no data if they're too small
- Check that predictions exist for all increments
- Verify HDF5 file contains all increments

## Integration with Presentation

These per-grain comparisons can be used in presentations to:
1. Show detailed model performance at grain level
2. Demonstrate which grain types are predicted well
3. Highlight boundary effects and complex grain interactions
4. Validate physics (elastic slopes, yield points)

## Related Scripts

- `analyze_per_grain_ss_curves_enhanced.py`: Per-grain analysis for single seeds (GT only)
- `evaluate_test.py`: Overall test set evaluation
- `visualize_prediction_with_props.py`: Property-stress visualization




