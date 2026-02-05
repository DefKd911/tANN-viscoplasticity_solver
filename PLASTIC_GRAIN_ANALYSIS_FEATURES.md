# Plastic Grain Behavior Analysis - New Features

## Overview

Enhanced the per-grain stress-strain analysis script to specifically analyze **plastic grain behavior** and test if the ML surrogate model predicts plastic grains well.

## New Features Added

### 1. Grain Behavior Classification

**Function**: `classify_grain_behavior()`

- Automatically classifies each grain as **'elastic'** or **'plastic'** based on stress-strain curve
- Criteria:
  - **Plastic**: Stress significantly exceeds elastic estimate OR stress exceeds yield stress (ξ₀)
  - **Elastic**: Stress follows elastic line closely
- Returns deviation percentage from elastic behavior

### 2. Microstructure Analysis Plot

**Function**: `plot_microstructure_with_properties()`

**9-Panel Layout**:
1. **Grain Behavior Map**: Color-coded map (Red=Plastic, Green=Elastic)
2. **Elastic Modulus (E)**: Spatial distribution with grain boundaries
3. **Poisson's Ratio (ν)**: Spatial distribution
4. **Initial Flow Stress (ξ₀)**: Spatial distribution
5. **Hardening Modulus (h₀)**: Spatial distribution
6. **Behavior Distribution**: Bar chart (Elastic vs Plastic count)
7. **ξ₀ Distribution by Behavior**: Histogram comparing elastic vs plastic grains
8. **E Distribution by Behavior**: Histogram comparing elastic vs plastic grains
9. **Summary Statistics**: Total grains, counts, plastic fraction

**Output**: `microstructure_analysis.png` per seed

### 3. Plastic vs Elastic Prediction Quality Comparison

**Function**: `plot_plastic_vs_elastic_prediction_quality()`

**2-Panel Comparison**:
- **Left Panel**: MAE (Mean Absolute Error) box plots for Elastic vs Plastic grains
- **Right Panel**: RMSE (Root Mean Square Error) box plots for Elastic vs Plastic grains
- Includes mean markers and statistics text
- Color-coded: Green (Elastic), Red (Plastic)

**Output**: `plastic_vs_elastic_prediction_quality.png` per seed

### 4. Enhanced Per-Grain Plots

**Updated**: `plot_per_grain_comparison()`

- Now includes behavior classification in title: `[ELASTIC]` or `[PLASTIC]`
- Helps identify which grains are being analyzed

## Output Structure

```
ML_EVAL/per_grain_test_analysis/
├── seed1099931136/
│   ├── microstructure_analysis.png          ← NEW: Properties + behavior map
│   ├── plastic_vs_elastic_prediction_quality.png  ← NEW: Error comparison
│   ├── grain_000_comparison.png            (with [ELASTIC] or [PLASTIC] label)
│   ├── grain_001_comparison.png
│   ├── ...
│   └── summary.json                        (includes behavior classification)
├── seed1098426137/
│   └── ...
└── overall_summary.json
```

## Enhanced Summary JSON

Now includes:
```json
{
  "grain_results": {
    "0": {
      "mae": 12.5,
      "rmse": 15.3,
      "behavior": "plastic",  ← NEW
      ...
    }
  },
  "grain_behaviors": {
    "0": "plastic",
    "1": "elastic",
    ...
  },
  "statistics": {
    "total_grains": 25,
    "elastic_count": 10,
    "plastic_count": 15,
    "elastic_mae_mean": 11.2,  ← NEW
    "plastic_mae_mean": 16.8,  ← NEW
    ...
  }
}
```

## Key Questions Answered

### ✅ Does the ML model predict plastic grains well?

**Analysis**:
- Compare MAE/RMSE for plastic vs elastic grains
- Check if plastic grains have significantly higher errors
- Identify patterns in property distributions

### ✅ What properties characterize plastic grains?

**Analysis**:
- Compare ξ₀ (yield stress) distributions
- Compare E (elastic modulus) distributions
- Visualize spatial distribution in microstructure

### ✅ Are there spatial patterns?

**Analysis**:
- Behavior map shows spatial clustering
- Property maps show correlation with behavior
- Grain boundaries may influence behavior

## Usage

Run the script as before - new features are automatically included:

```bash
python analyze_test_per_grain_ss_curves.py \
    --seeds 1099931136,1098426137,1098322009,1098694165,1102020331 \
    --max-grains 20
```

## Interpretation Guide

### Good Plastic Prediction
- **Plastic grains MAE** similar to or only slightly higher than elastic grains
- **Plastic grain curves** follow ground truth well
- **No systematic bias** in plastic grain predictions

### Poor Plastic Prediction
- **Plastic grains MAE** significantly higher than elastic grains (>5-10 MPa difference)
- **Plastic grain curves** deviate from ground truth
- **Systematic underestimation** or **overestimation** of plastic response

### Property Insights
- **Low ξ₀** → More likely to be plastic (yield early)
- **High ξ₀** → More likely to be elastic (don't yield at 0.4% strain)
- **Spatial clustering** → Neighboring grains may have similar behavior

## Example Findings

Based on typical results:

1. **Plastic grains** often have:
   - Lower ξ₀ (yield stress)
   - Similar or slightly higher prediction errors
   - More complex stress-strain curves

2. **Elastic grains** often have:
   - Higher ξ₀ (don't yield)
   - Lower prediction errors
   - Simpler stress-strain curves (linear)

3. **Model Performance**:
   - If plastic MAE ≈ elastic MAE → Model handles plasticity well ✅
   - If plastic MAE >> elastic MAE → Model struggles with plasticity ⚠️

## Integration with Presentation

These new plots can be used to:

1. **Demonstrate model capability**: Show that plastic grains are predicted well
2. **Identify limitations**: Highlight if plastic grains have higher errors
3. **Property-behavior correlation**: Show how properties relate to behavior
4. **Spatial analysis**: Visualize microstructure-property-behavior relationships

## Files Modified

- `analyze_test_per_grain_ss_curves.py`: Added 3 new functions and integrated into main loop
- Enhanced summary JSON with behavior classification
- New plot outputs per seed

## Dependencies

- `scipy` (optional but recommended for better boundary detection)
- All existing dependencies remain the same

---

**Status**: ✅ Complete and ready to use!




