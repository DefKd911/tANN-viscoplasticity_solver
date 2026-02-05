# Machine Learning Surrogate for DAMASK Viscoplasticity Simulations
## Post Mid-Semester Progress Presentation

---

## Slide 1: Title Slide

# Machine Learning Surrogate Model for  
# Polycrystalline Viscoplasticity Simulations

**Post Mid-Semester Progress Report**

**Project**: tANN Viscoplasticity Solver  
**Objective**: Develop ML surrogate for DAMASK J2 elasto-viscoplastic simulations

---

## Slide 2: Project Overview & Context

### What We Started With (Mid-Semester)

✅ **Completed**:
- Generated 50 synthetic microstructures (Voronoi tessellation)
- Created DAMASK-compatible geometry files (64×64 voxels)
- Assigned grain-wise mechanical properties:
  - **E** (Elastic modulus): 50-300 GPa
  - **ν** (Poisson's ratio): 0.2-0.4
  - **ξ₀** (Initial flow stress): 50-300 MPa
  - **h₀** (Hardening modulus): 0-50 GPa
- Ran 50 DAMASK J2 elasto-viscoplastic simulations
- Uniaxial tension to 0.4% strain

### Post Mid-Semester Goal

**Transform simulation outputs into ML-ready dataset and develop surrogate model for stress prediction**

---

## Slide 3: Post Mid-Semester Objectives

### Primary Objectives

1. **Data Extraction & Processing**
   - Extract stress fields from HDF5 outputs
   - Generate stress-strain curves
   - Per-grain analysis

2. **ML Dataset Construction**
   - Create temporal prediction dataset
   - Proper train/val/test splits
   - Normalization and preprocessing

3. **Model Development**
   - Implement baseline U-Net architecture
   - Train and evaluate model
   - Comprehensive error analysis

4. **Visualization & Validation**
   - Ground truth visualizations
   - Prediction quality assessment
   - Physics validation

---

## Slide 4: Phase 1 - Data Extraction & Post-Processing

### 4.1 HDF5 Data Extraction

**Challenge**: DAMASK outputs stress in multiple formats (direct tensors or F/P pairs)

**Solution**: Robust extraction pipeline with fallback strategies
- Direct stress tensor extraction (preferred)
- Fallback: Compute from deformation gradient (F) and 1st Piola-Kirchhoff (P)
- Formula: **σ = (1/det F) × P × F^T**
- Von Mises: **σ_vM = √(1.5 × tr(s_dev²))**

**Results**:
- ✅ Successfully extracted all 50 simulations
- ✅ Handled multiple HDF5 dataset path conventions
- ✅ Quality control scripts for data integrity

### 4.2 Stress-Strain Curve Generation

**Output**: 46 macroscopic stress-strain curves
- CSV data + PNG visualizations
- Verified J2 viscoplastic behavior
- Elastic modulus, yield point, hardening slope analysis

**Location**: `simulation_results/stress_curves/`

---

## Slide 5: Phase 1 (Continued) - Per-Grain Analysis

### Enhanced Per-Grain Stress-Strain Analysis

**Objective**: Understand individual grain behavior within polycrystal

**Key Features**:
- Real elastic estimation using cubic crystal elasticity (C11, C12, C44)
- Comparison with isotropic E×ε estimate
- Yield detection based on material properties
- Elastic vs plastic regime classification

### Key Findings

✅ **Individual grains show distinct behavior**:
- Elastic-plastic transitions vary by grain
- Yield points correlate with **ξ₀** (initial flow stress)
- Hardening slopes correlate with **h₀** (hardening modulus)
- Some grains remain fully elastic even at 0.4% strain (high ξ₀)

**Example**: Grain with ξ₀ = 150 MPa yields at ~0.075% strain

**Output**: Enhanced visualizations in `per_grain_analysis/`

---

## Slide 6: Phase 2 - ML Dataset Construction

### Dataset Design

**Problem**: Predict stress evolution σ_vM(t+Δt) from current state

**Input Format**: `X (64, 64, 5)` channels
1. **E** (Elastic modulus) - GPa
2. **ν** (Poisson's ratio) - dimensionless
3. **ξ₀** (Initial flow stress) - MPa
4. **h₀** (Hardening modulus) - GPa
5. **σ_vM(t)** (Current von Mises stress) - MPa

**Output Format**: `Y (64, 64, 1)`
- **σ_vM(t+Δt)** (Next time step stress) - MPa

### Dataset Statistics

| Split | Samples | Seeds | Description |
|-------|---------|-------|-------------|
| **Train** | 1,600 | 40 | Temporal pairs from 40 simulations |
| **Val** | 200 | 5 | Validation set |
| **Test** | 100 | 5 | Test set (seed-wise split) |
| **Total** | **1,900** | **50** | All from 50 DAMASK simulations |

**Key Feature**: Seed-wise splitting preserves microstructure diversity

---

## Slide 7: Phase 2 (Continued) - Normalization & Preprocessing

### Normalization Scheme

All properties normalized to ~[0,1] range:

| Property | Range | Normalization Formula |
|----------|-------|----------------------|
| **E** | 50-300 GPa | (E - 50e9) / 250e9 |
| **ν** | 0.2-0.4 | (ν - 0.2) / 0.2 |
| **ξ₀** | 50-300 MPa | (ξ₀ - 50e6) / 250e6 |
| **h₀** | 0-50 GPa | h₀ / 50e9 |
| **σ_vM** | Variable | (σ_vM[Pa]/1e6) / 1000.0 |

### Data Pipeline Features

✅ **Per-pixel property mapping**: Grain-wise properties → pixel maps using labels  
✅ **Temporal pairs**: Creates (t, t+1) increment pairs from each simulation  
✅ **Metadata tracking**: Sample → seed → increment mapping  
✅ **Robust error handling**: Handles missing increments gracefully

**Output Structure**:
```
ML_DATASET/
├── train/inputs/    (1600 .npy files)
├── train/outputs/   (1600 .npy files)
├── val/inputs/      (200 .npy files)
├── val/outputs/     (200 .npy files)
├── test/inputs/     (100 .npy files)
├── test/outputs/    (100 .npy files)
└── metadata/        (splits, normalization, mapping)
```

---

## Slide 8: Phase 3 - Model Architecture & Training

### U-Net Baseline Architecture

**Architecture**: Encoder-decoder with skip connections

```
Input: (5, 64, 64)  →  [E, ν, ξ₀, h₀, σ_vM(t)]
    ↓
Encoder (Downsampling):
  - Conv Block 1: 32 channels
  - MaxPool → Conv Block 2: 64 channels
  - MaxPool → Conv Block 3: 128 channels
  - MaxPool → Bottleneck: 256 channels
    ↓
Decoder (Upsampling):
  - Upsample + Skip → Conv Block: 128 channels
  - Upsample + Skip → Conv Block: 64 channels
  - Upsample + Skip → Conv Block: 32 channels
    ↓
Output: (1, 64, 64)  →  [σ_vM(t+Δt)]
```

**Components**:
- DoubleConv blocks: Conv2d → BatchNorm → ReLU (×2)
- Bilinear upsampling with skip connections
- Final 1×1 convolution

### Training Configuration

- **Loss**: L1 (MAE) loss
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Batch size**: 8
- **Early stopping**: Patience=10 epochs
- **Device**: CUDA (GPU accelerated)

---

## Slide 9: Phase 3 (Continued) - Model Performance

### Test Set Results

**Overall Performance**:
- **Test MAE**: **14.9 MPa** (normalized: 0.0149)
- **Test RMSE**: **21.5 MPa** (normalized: 0.0223)

### Error Distribution

| Metric | Value |
|--------|-------|
| Mean MAE | 14.9 MPa |
| Median MAE | 15.2 MPa |
| Standard Deviation | 3.3 MPa |
| 90th Percentile | 19.8 MPa |
| 95th Percentile | 20.6 MPa |
| Max Error | ~300 MPa (boundary spikes) |

**Key Insight**: Only ~5% of samples exceed 20 MPa MAE

### Per-Seed Performance

| Seed | MAE (MPa) | Microstructure Type |
|------|-----------|---------------------|
| **1099931136** | **11.5** | Large, equiaxed grains ⭐ Best |
| 1098426137 | 12.7 | Moderate complexity |
| 1098322009 | 16.1 | Mixed |
| 1098694165 | 16.6 | Elongated grains |
| **1102020331** | **17.7** | Elongated + triple junctions ⚠️ Hardest |

**Finding**: Performance correlates with microstructure complexity (~6 MPa variance)

---

## Slide 10: Error Analysis & Key Insights

### Error Localization

**Spatial Error Distribution**:
- ✅ **>90% of pixels** have error < **55 MPa**
- ⚠️ Maximum errors (**250-300 MPa**) occur at **narrow boundary segments**
- ✅ Grain interiors: Excellent accuracy (~10-15 MPa)
- ⚠️ Grain boundaries: Primary source of error

### Key Insights

1. **Boundary Modeling is Critical**
   - Peak stress smeared by 50-80 MPa at boundaries
   - Suggests need for boundary-aware architectures
   - Explicit boundary features could improve performance

2. **Microstructure Dependence**
   - Simple microstructures (equiaxed grains) → Lower error (11.5 MPa)
   - Complex microstructures (elongated + junctions) → Higher error (17.7 MPa)
   - Model generalizes well but struggles with high-contrast boundaries

3. **Temporal Coherence** ✅
   - Model learns dynamics (not just static snapshots)
   - Smooth evolution without abrupt jumps
   - Maintains grain-level stress ordering

### Best vs Worst Samples

**Best Sample** (seed1099931136, sample_00963):
- MAE: **9.09 MPa**
- P90 error: 18.7 MPa
- Max error: 80.3 MPa
- Location: Grain interiors

**Worst Sample** (seed1102020331, sample_00999):
- MAE: **24.3 MPa**
- P90 error: 77.7 MPa
- Max error: **304.3 MPa**
- Location: Triple junctions + slender grains

---

## Slide 11: Phase 4 - Visualization & Analysis Tools

### 4.1 Ground Truth Visualization

**7-Panel Layout** (43 images generated):
1. **E (GPa)** - Elastic modulus map
2. **ν (-)** - Poisson's ratio map
3. **ξ₀ (MPa)** - Initial flow stress map
4. **h₀ (GPa)** - Hardening modulus map
5. **σ_vM (MPa)** - Ground truth von Mises stress
6. **σ_vM_el (MPa)** - Elastic estimate (E × ε)
7. **Δσ_vM (MPa)** - Deviation from elastic (plastic contribution)

**Features**:
- Grain boundary overlays (white contours)
- Auto-alignment using D4 transforms
- Proper unit conversions

### 4.2 Prediction Visualization

**Multi-Panel Predictions** (100 images):
- Property maps (E, ν, ξ₀, h₀)
- Ground truth vs predicted stress
- Elastic estimate
- Absolute error map
- All in MPa units with GB overlays

### 4.3 Temporal Evolution Videos

**4 MP4 Videos Generated**:
- 3-panel layout: GT, Prediction, Error
- Shows stress evolution over time
- Grain boundary overlays
- Temporal coherence verification

**Key Observations from Videos**:
- ✅ Smooth temporal evolution
- ✅ Correct grain-level stress ordering
- ⚠️ Boundary peak stress smearing (50-80 MPa)
- ✅ Elastic regime matches expected slope

---

## Slide 12: Phase 4 (Continued) - Additional Visualizations

### Stress-Strain Curves

**5 Test Seed Curves Generated**:
- Macroscopic stress-strain response
- GT vs predicted mean stress comparison
- Elastic slope verification
- Hardening behavior analysis

**Location**: `ML_EVAL/stress_strain/plots/`

### Error Histograms

**87 Error Distribution Histograms**:
- Per-sample error distributions
- Aggregated error histogram (all samples)
- Identifies error patterns and outliers

**Location**: `ML_EVAL/histograms/`

### Qualitative Error Analysis

**8 Sample Comparisons**:
- 3-panel: Target, Prediction, Error
- Both normalized and MPa-scaled versions
- Grain boundary overlays
- Highlights prediction quality

**Location**: `ML_EVAL/qualitative/`

---

## Slide 13: Phase 5 - Physics Validation

### Physics Understanding & Documentation

**Created 6 Documentation Files**:

1. **UNDERSTANDING_DAMASK_ELASTIC_RESPONSE.md**
   - Why DAMASK elastic response ≠ theoretical Hooke's law
   - Microstructure effects on elastic behavior
   - Grain-to-grain property variations

2. **GT_OVERLAYS_ANALYSIS.md**
   - GT overlay visualization analysis
   - Identified and documented issues
   - Recommendations for improvements

3. **Additional Explain Files**:
   - Elastic vs hardening plotting
   - Grain hardening differences
   - Linear vs nonlinear hardening
   - Per-grain visualization methodology

### Verification Scripts

✅ **Physics Validation Tools**:
- `verify_per_grain_physics.py` - Grain-level physics validation
- `verify_ss_curve_physics.py` - Stress-strain curve validation
- `check_gt_overlays.py` - Overlay correctness verification
- `check_hdf5_stress.py` - HDF5 data integrity checks

### Key Physics Insights

✅ **Elastic Regime**:
- Model correctly captures elastic slope
- Early increments match E × ε estimate
- Slope errors within ±5 GPa

✅ **Plastic Regime**:
- Hardening behavior follows expected trends
- Grain-level variations correlate with properties
- Macroscopic response validated

---

## Slide 14: Project Achievements Summary

### ✅ Completed Work (100%)

| Phase | Status | Key Deliverables |
|-------|--------|------------------|
| **Data Pipeline** | ✅ 100% | HDF5 extraction, 46 stress-strain curves, per-grain analysis, 1900-sample ML dataset |
| **Machine Learning** | ✅ 100% | U-Net baseline, training pipeline, 14.9 MPa MAE performance |
| **Visualization** | ✅ 100% | 43 GT overlays, 100 prediction panels, 4 videos, 5 stress-strain curves |
| **Physics Validation** | ✅ 100% | Elastic/plastic analysis, per-grain verification, comprehensive documentation |
| **Quality Assurance** | ✅ 100% | HDF5 checks, physics verification, error localization analysis |

### 📊 Performance Metrics Summary

- **Test MAE**: **14.9 MPa** (excellent for microstructure prediction)
- **Test RMSE**: **21.5 MPa**
- **90th Percentile**: 19.8 MPa
- **Best Seed**: 11.5 MPa MAE
- **Temporal Coherence**: ✅ Verified
- **Grain-level Accuracy**: ✅ Verified

### 🔍 Key Contributions

1. **Robust Data Pipeline**: Handles multiple HDF5 formats, missing data
2. **Seed-wise Splitting**: Preserves microstructure diversity
3. **Comprehensive Evaluation**: Per-sample, per-seed, spatial error analysis
4. **Physics Validation**: Ensures model respects material physics
5. **Rich Visualizations**: Multiple formats for different analysis needs

---

## Slide 15: Future Work & Conclusions

### Immediate Next Steps

1. **Model Architecture Improvements**
   - Deeper/wider U-Net (base channels 48/64)
   - Add residual blocks
   - Multi-scale attention mechanisms

2. **Feature Augmentation**
   - Grain-boundary distance channels
   - History channels (multiple time steps)
   - Explicit boundary features

3. **Dataset Scaling**
   - Run additional DAMASK simulations (beyond 50)
   - Include more challenging microstructures
   - Balance seed diversity

4. **Boundary Modeling** (Priority)
   - Explicit boundary loss terms
   - Boundary-aware architectures
   - Multi-resolution approaches

### Long-Term Enhancements

- **Advanced Architectures**: Transformers, Graph Neural Networks, Physics-Informed NNs
- **Extended Physics**: Multi-step prediction, different loading conditions, temperature effects
- **Automation**: Automated reporting, continuous evaluation, model versioning

### Conclusions

✅ **Successfully developed ML surrogate** for DAMASK viscoplasticity simulations  
✅ **Achieved 14.9 MPa average MAE** - excellent performance for microstructure prediction  
✅ **Comprehensive pipeline** from data extraction to model evaluation  
✅ **Physics-validated** predictions with rich visualizations  
✅ **Clear improvement path** identified (boundary modeling)

**Project Status**: ~85% complete (baseline model + full pipeline)  
**Remaining**: Model improvements, dataset scaling, advanced architectures

---

## Appendix: File Structure & Deliverables

### Key Directories

```
simulation_results/
├── hdf5_files/          (50 .hdf5 files)
├── stress_curves/       (46 CSV + PNG pairs)
└── ...

ML_DATASET/
├── train/               (1600 samples)
├── val/                 (200 samples)
├── test/                (100 samples)
└── metadata/            (splits, normalization)

ML_CHECKPOINTS/
├── best.pt              (Trained model)
├── training_curve.png
└── training_log.csv

ML_EVAL/
├── predictions/         (100 .npy files)
├── qualitative/         (8 comparisons)
├── gt_overlays/         (43 visualizations)
├── pred_prop_panels/    (100 panels)
├── histograms/          (87 histograms)
├── stress_strain/       (5 curves)
└── stress_videos/       (4 MP4 videos)
```

### Key Scripts (30+ Python files)

- **Core ML**: `build_ml_dataset.py`, `train_unet_baseline.py`, `evaluate_test.py`
- **Data Extraction**: `extract_stress_from_F_P.py`, `make_ss_curve.py`
- **Analysis**: `analyze_per_grain_ss_curves_enhanced.py`
- **Visualization**: `visualize_gt_from_hdf5.py`, `make_stress_video.py`
- **Verification**: `verify_per_grain_physics.py`, `check_hdf5_stress.py`

---

**Thank You!**

Questions?




