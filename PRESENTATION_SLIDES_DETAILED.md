# Presentation Slides: Post Mid-Semester Progress
## Machine Learning Surrogate for DAMASK Viscoplasticity Simulations

---

## SLIDE 1: TITLE SLIDE

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Machine Learning Surrogate Model for                     ║
║     Polycrystalline Viscoplasticity Simulations              ║
║                                                              ║
║                    Post Mid-Semester                         ║
║                    Progress Report                           ║
║                                                              ║
║                                                              ║
║     Project: tANN Viscoplasticity Solver                     ║
║     Objective: ML Surrogate for DAMASK J2                    ║
║                Elasto-Viscoplastic Simulations                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Visual Elements to Add**:
- [ ] Project logo (if available)
- [ ] DAMASK simulation visualization (microstructure image)
- [ ] Date and presenter name

---

## SLIDE 2: PROJECT OVERVIEW & CONTEXT

### What We Accomplished (Mid-Semester)

**✅ Completed Foundation**:
- Generated **50 synthetic microstructures** using Voronoi tessellation
- Created **DAMASK-compatible geometry** files (64×64 voxels)
- Assigned **grain-wise mechanical properties**:
  - **E** (Elastic modulus): 50-300 GPa
  - **ν** (Poisson's ratio): 0.2-0.4
  - **ξ₀** (Initial flow stress): 50-300 MPa
  - **h₀** (Hardening modulus): 0-50 GPa
- Ran **50 DAMASK J2 elasto-viscoplastic simulations**
- Uniaxial tension loading to **0.4% strain**

### Post Mid-Semester Objective

**🎯 Transform simulation outputs into ML-ready dataset and develop surrogate model for stress prediction**

**Visual Elements to Add**:
- [ ] Voronoi microstructure example image
- [ ] Property distribution histograms (E, ν, ξ₀, h₀)
- [ ] Sample stress-strain curve from DAMASK

---

## SLIDE 3: POST MID-SEMESTER OBJECTIVES

### Primary Objectives

**1. Data Extraction & Processing** 📊
   - Extract stress fields from HDF5 outputs
   - Generate macroscopic stress-strain curves
   - Per-grain behavior analysis

**2. ML Dataset Construction** 🗂️
   - Create temporal prediction dataset
   - Proper train/val/test splits (seed-wise)
   - Normalization and preprocessing

**3. Model Development** 🤖
   - Implement baseline U-Net architecture
   - Train and evaluate model
   - Comprehensive error analysis

**4. Visualization & Validation** 📈
   - Ground truth visualizations
   - Prediction quality assessment
   - Physics validation

**Visual Elements to Add**:
- [ ] Workflow diagram showing 4 phases
- [ ] Timeline or Gantt chart

---

## SLIDE 4: PHASE 1 - DATA EXTRACTION

### Challenge: Multiple HDF5 Formats

**Problem**: DAMASK outputs stress in different formats
- Direct stress tensors (preferred)
- Deformation gradient (F) + 1st Piola-Kirchhoff (P) pairs

### Solution: Robust Extraction Pipeline

**Strategy**: Multi-level fallback approach
1. **Primary**: Extract direct stress tensor
2. **Fallback**: Compute from F and P
   - Formula: **σ = (1/det F) × P × F^T**
   - Von Mises: **σ_vM = √(1.5 × tr(s_dev²))**

**Results**:
- ✅ Successfully extracted **all 50 simulations**
- ✅ Handled multiple HDF5 dataset path conventions
- ✅ Quality control scripts for data integrity

**Visual Elements to Add**:
- [ ] Flowchart of extraction pipeline
- [ ] Sample HDF5 structure diagram
- [ ] Before/after extraction comparison

---

## SLIDE 5: PHASE 1 - STRESS-STRAIN CURVES & PER-GRAIN ANALYSIS

### Macroscopic Stress-Strain Curves

**Output**: **46 stress-strain curves** generated
- CSV data + PNG visualizations
- Verified J2 viscoplastic behavior
- Elastic modulus, yield point, hardening slope analysis

### Enhanced Per-Grain Analysis

**Key Features**:
- Real elastic estimation using cubic crystal elasticity
- Yield detection based on material properties
- Elastic vs plastic regime classification

**Key Findings**:
- ✅ Individual grains show distinct elastic-plastic transitions
- ✅ Yield points vary significantly (correlate with **ξ₀**)
- ✅ Hardening slopes correlate with **h₀**
- ✅ Some grains remain fully elastic at 0.4% strain

**Example**: Grain with ξ₀ = 150 MPa → yields at ~0.075% strain

**Visual Elements to Add**:
- [ ] Sample macroscopic stress-strain curve
- [ ] Per-grain stress-strain curves (multiple grains)
- [ ] Elastic vs plastic regime visualization
- [ ] Yield point distribution histogram

---

## SLIDE 6: PHASE 2 - ML DATASET DESIGN

### Problem Formulation

**Task**: Predict stress evolution **σ_vM(t+Δt)** from current state

### Input-Output Format

**Input**: `X (64, 64, 5)` channels
```
Channel 1: E  (Elastic modulus)        [GPa]
Channel 2: ν  (Poisson's ratio)        [dimensionless]
Channel 3: ξ₀ (Initial flow stress)    [MPa]
Channel 4: h₀ (Hardening modulus)      [GPa]
Channel 5: σ_vM(t) (Current stress)    [MPa]
```

**Output**: `Y (64, 64, 1)`
```
Channel 1: σ_vM(t+Δt) (Next time step) [MPa]
```

### Dataset Statistics

| Split | Samples | Seeds | Description |
|-------|---------|-------|-------------|
| **Train** | **1,600** | 40 | Temporal pairs from 40 simulations |
| **Val** | **200** | 5 | Validation set |
| **Test** | **100** | 5 | Test set (seed-wise split) |
| **Total** | **1,900** | **50** | All from 50 DAMASK simulations |

**Key Feature**: ✅ **Seed-wise splitting** preserves microstructure diversity

**Visual Elements to Add**:
- [ ] Input tensor visualization (5-channel image)
- [ ] Output tensor visualization (1-channel image)
- [ ] Dataset split pie chart
- [ ] Sample count bar chart

---

## SLIDE 7: PHASE 2 - NORMALIZATION & PREPROCESSING

### Normalization Scheme

All properties normalized to ~[0,1] range for stable training:

| Property | Physical Range | Normalization Formula |
|----------|----------------|----------------------|
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

**Visual Elements to Add**:
- [ ] Before/after normalization comparison
- [ ] Normalized property distribution plots
- [ ] Data pipeline flowchart
- [ ] Sample temporal pair visualization

---

## SLIDE 8: PHASE 3 - MODEL ARCHITECTURE

### U-Net Baseline Architecture

**Type**: Encoder-decoder with skip connections

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT (5, 64, 64)                    │
│              [E, ν, ξ₀, h₀, σ_vM(t)]                   │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
    ENCODER                         DECODER
    (Downsampling)                  (Upsampling)
        │                               │
    Conv Block 1 (32)              Upsample + Skip
        │                               │
    MaxPool → Conv Block 2 (64)    Conv Block (128)
        │                               │
    MaxPool → Conv Block 3 (128)   Upsample + Skip
        │                               │
    MaxPool → Bottleneck (256)     Conv Block (64)
        │                               │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │         OUTPUT (1, 64, 64)    │
        │         [σ_vM(t+Δt)]          │
        └───────────────────────────────┘
```

### Training Configuration

- **Loss Function**: L1 (MAE) loss
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Batch Size**: 8
- **Early Stopping**: Patience=10 epochs
- **Device**: CUDA (GPU accelerated)

**Visual Elements to Add**:
- [ ] U-Net architecture diagram (detailed)
- [ ] Training curve (loss vs epoch)
- [ ] Model parameter count

---

## SLIDE 9: PHASE 3 - MODEL PERFORMANCE

### Test Set Results

**Overall Performance Metrics**:

```
╔═══════════════════════════════════════════╗
║  Test MAE:  14.9 MPa  (normalized: 0.0149)║
║  Test RMSE: 21.5 MPa  (normalized: 0.0223)║
╚═══════════════════════════════════════════╝
```

### Error Distribution Statistics

| Metric | Value |
|--------|-------|
| **Mean MAE** | **14.9 MPa** |
| Median MAE | 15.2 MPa |
| Standard Deviation | 3.3 MPa |
| **90th Percentile** | **19.8 MPa** |
| 95th Percentile | 20.6 MPa |
| Max Error | ~300 MPa (boundary spikes) |

**Key Insight**: ✅ Only ~5% of samples exceed 20 MPa MAE

### Per-Seed Performance

| Seed | MAE (MPa) | Microstructure Type | Status |
|------|-----------|---------------------|--------|
| **1099931136** | **11.5** | Large, equiaxed grains | ⭐ Best |
| 1098426137 | 12.7 | Moderate complexity | ✅ Good |
| 1098322009 | 16.1 | Mixed | ⚠️ Moderate |
| 1098694165 | 16.6 | Elongated grains | ⚠️ Moderate |
| **1102020331** | **17.7** | Elongated + triple junctions | ⚠️ Hardest |

**Finding**: Performance correlates with microstructure complexity (~6 MPa variance)

**Visual Elements to Add**:
- [ ] Error distribution histogram
- [ ] Per-seed MAE bar chart
- [ ] Box plot of errors by seed
- [ ] Cumulative error distribution

---

## SLIDE 10: ERROR ANALYSIS & KEY INSIGHTS

### Spatial Error Localization

**Error Distribution**:
- ✅ **>90% of pixels** have error < **55 MPa**
- ⚠️ Maximum errors (**250-300 MPa**) occur at **narrow boundary segments**
- ✅ Grain interiors: Excellent accuracy (~10-15 MPa)
- ⚠️ Grain boundaries: Primary source of error

### Key Insights

**1. Boundary Modeling is Critical** 🎯
   - Peak stress smeared by 50-80 MPa at boundaries
   - Suggests need for boundary-aware architectures
   - Explicit boundary features could improve performance

**2. Microstructure Dependence** 📊
   - Simple microstructures (equiaxed) → Lower error (11.5 MPa)
   - Complex microstructures (elongated + junctions) → Higher error (17.7 MPa)
   - Model generalizes well but struggles with high-contrast boundaries

**3. Temporal Coherence** ✅
   - Model learns dynamics (not just static snapshots)
   - Smooth evolution without abrupt jumps
   - Maintains grain-level stress ordering

### Best vs Worst Samples

**Best Sample** (seed1099931136, sample_00963):
- MAE: **9.09 MPa** ⭐
- P90 error: 18.7 MPa
- Max error: 80.3 MPa
- Location: Grain interiors

**Worst Sample** (seed1102020331, sample_00999):
- MAE: **24.3 MPa** ⚠️
- P90 error: 77.7 MPa
- Max error: **304.3 MPa**
- Location: Triple junctions + slender grains

**Visual Elements to Add**:
- [ ] Spatial error map (heatmap)
- [ ] Error histogram (pixel-wise)
- [ ] Best vs worst sample comparison
- [ ] Boundary error visualization

---

## SLIDE 11: PHASE 4 - VISUALIZATION TOOLS

### 4.1 Ground Truth Visualization (7-Panel Layout)

**43 GT Overlay Images Generated**:

```
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  E   │  ν   │  ξ₀  │  h₀  │ σ_vM │σ_vM_el│ Δσ_vM│
│ (GPa)│  (-) │(MPa) │(GPa) │(MPa) │(MPa) │(MPa) │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

**Features**:
- Grain boundary overlays (white contours)
- Auto-alignment using D4 transforms
- Proper unit conversions

### 4.2 Prediction Visualization

**100 Multi-Panel Predictions**:
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

**Key Observations**:
- ✅ Smooth temporal evolution
- ✅ Correct grain-level stress ordering
- ⚠️ Boundary peak stress smearing (50-80 MPa)
- ✅ Elastic regime matches expected slope

**Visual Elements to Add**:
- [ ] Sample 7-panel GT overlay image
- [ ] Sample prediction panel
- [ ] Video frame screenshot
- [ ] Temporal evolution comparison

---

## SLIDE 12: PHASE 4 - ADDITIONAL VISUALIZATIONS

### Stress-Strain Curves

**5 Test Seed Curves Generated**:
- Macroscopic stress-strain response
- GT vs predicted mean stress comparison
- Elastic slope verification
- Hardening behavior analysis

### Error Histograms

**87 Error Distribution Histograms**:
- Per-sample error distributions
- Aggregated error histogram (all samples)
- Identifies error patterns and outliers

### Qualitative Error Analysis

**8 Sample Comparisons**:
- 3-panel: Target, Prediction, Error
- Both normalized and MPa-scaled versions
- Grain boundary overlays
- Highlights prediction quality

**Visual Elements to Add**:
- [ ] Sample stress-strain curve (GT vs Predicted)
- [ ] Aggregated error histogram
- [ ] Sample qualitative comparison (3-panel)
- [ ] Error distribution across samples

---

## SLIDE 13: PHASE 5 - PHYSICS VALIDATION

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

**Visual Elements to Add**:
- [ ] Elastic vs plastic regime comparison
- [ ] Physics validation checklist
- [ ] Slope error analysis plot

---

## SLIDE 14: PROJECT ACHIEVEMENTS SUMMARY

### ✅ Completed Work (100%)

| Phase | Status | Key Deliverables |
|-------|--------|------------------|
| **Data Pipeline** | ✅ 100% | HDF5 extraction, 46 stress-strain curves, per-grain analysis, 1900-sample ML dataset |
| **Machine Learning** | ✅ 100% | U-Net baseline, training pipeline, **14.9 MPa MAE** performance |
| **Visualization** | ✅ 100% | 43 GT overlays, 100 prediction panels, 4 videos, 5 stress-strain curves |
| **Physics Validation** | ✅ 100% | Elastic/plastic analysis, per-grain verification, comprehensive documentation |
| **Quality Assurance** | ✅ 100% | HDF5 checks, physics verification, error localization analysis |

### 📊 Performance Metrics Summary

```
╔═══════════════════════════════════════════════╗
║  Test MAE:        14.9 MPa                    ║
║  Test RMSE:       21.5 MPa                    ║
║  90th Percentile: 19.8 MPa                    ║
║  Best Seed:       11.5 MPa MAE                ║
║  Temporal Coherence: ✅ Verified              ║
║  Grain-level Accuracy: ✅ Verified            ║
╚═══════════════════════════════════════════════╝
```

### 🔍 Key Contributions

1. **Robust Data Pipeline**: Handles multiple HDF5 formats, missing data
2. **Seed-wise Splitting**: Preserves microstructure diversity
3. **Comprehensive Evaluation**: Per-sample, per-seed, spatial error analysis
4. **Physics Validation**: Ensures model respects material physics
5. **Rich Visualizations**: Multiple formats for different analysis needs

**Visual Elements to Add**:
- [ ] Progress bar showing 100% completion
- [ ] Key metrics dashboard
- [ ] Achievement checklist

---

## SLIDE 15: FUTURE WORK & CONCLUSIONS

### Immediate Next Steps

**1. Model Architecture Improvements** 🏗️
   - Deeper/wider U-Net (base channels 48/64)
   - Add residual blocks
   - Multi-scale attention mechanisms

**2. Feature Augmentation** 📈
   - Grain-boundary distance channels
   - History channels (multiple time steps)
   - Explicit boundary features

**3. Dataset Scaling** 📊
   - Run additional DAMASK simulations (beyond 50)
   - Include more challenging microstructures
   - Balance seed diversity

**4. Boundary Modeling** (Priority) 🎯
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

**Visual Elements to Add**:
- [ ] Roadmap diagram
- [ ] Future work timeline
- [ ] Project completion percentage

---

## SLIDE 16: APPENDIX - FILE STRUCTURE

### Key Deliverables

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

## SLIDE 17: THANK YOU

```
╔═══════════════════════════════════════════════╗
║                                               ║
║              Thank You!                       ║
║                                               ║
║            Questions?                         ║
║                                               ║
╚═══════════════════════════════════════════════╝
```

**Contact Information**:
- [ ] Email
- [ ] GitHub repository link
- [ ] Project website (if available)

---

## PRESENTATION NOTES

### Slide Timing Guide (15-20 minutes total)

- **Slide 1-2**: Introduction (2 min)
- **Slide 3**: Objectives (1 min)
- **Slide 4-5**: Data Extraction (3 min)
- **Slide 6-7**: Dataset Construction (3 min)
- **Slide 8-9**: Model & Performance (4 min)
- **Slide 10**: Error Analysis (2 min)
- **Slide 11-12**: Visualizations (2 min)
- **Slide 13**: Physics Validation (1 min)
- **Slide 14**: Achievements (1 min)
- **Slide 15**: Future Work (1 min)
- **Slide 16-17**: Appendix & Closing (1 min)

### Key Points to Emphasize

1. **14.9 MPa MAE** - This is excellent performance for microstructure prediction
2. **Comprehensive pipeline** - From raw HDF5 to trained model
3. **Physics validation** - Model respects material physics
4. **Clear improvement path** - Boundary modeling identified as priority

### Visual Recommendations

- Use consistent color scheme throughout
- Highlight key numbers (14.9 MPa, 1900 samples, etc.)
- Include actual images from your results folders
- Use arrows/flowcharts to show data flow
- Add comparison plots (GT vs Predicted)

---

**END OF PRESENTATION**




