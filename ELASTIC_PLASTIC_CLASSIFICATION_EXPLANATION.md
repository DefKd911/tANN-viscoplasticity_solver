# Elastic vs Plastic Grain Classification - Detailed Explanation

## Overview

The script automatically classifies each grain as **ELASTIC** or **PLASTIC** based on its stress-strain behavior from DAMASK simulations. This classification helps analyze whether the ML model predicts plastic grain behavior well.

---

## Classification Criteria

A grain is classified as **PLASTIC** if **ANY** of the following conditions are met:

### Criterion 1: Yield Stress Criterion

**Formula**: `max(σ_vM) > 1.1 × ξ₀`

**Physical Meaning**:
- The grain's maximum von Mises stress exceeds its yield stress (ξ₀) by more than 10%
- The 10% margin accounts for numerical noise and small variations
- If this condition is true, the grain has **yielded** and entered the plastic regime

**Example**:
- Grain with ξ₀ = 200 MPa
- If max stress = 150 MPa → **ELASTIC** (hasn't reached yield)
- If max stress = 250 MPa → **PLASTIC** (exceeded yield stress)

### Criterion 2: Deviation from Elastic Line Criterion

**Formula**: `max(|σ_vM - σ_elastic| / σ_elastic) > yield_threshold`

Where:
- `σ_elastic = E × ε` (Hooke's law)
- `yield_threshold = 0.1` (default: 10% deviation)

**Physical Meaning**:
- The grain's stress significantly deviates from the linear elastic prediction
- If deviation exceeds 10%, the grain is showing **non-linear behavior** (plasticity)
- This captures grains that may have yielded even if they don't exceed ξ₀ (due to microstructure effects)

**Example**:
- Grain with E = 200 GPa, ε = 0.002 (0.2% strain)
- Elastic prediction: σ = 200 GPa × 0.002 = 400 MPa
- If actual stress = 420 MPa → deviation = 5% → **ELASTIC**
- If actual stress = 450 MPa → deviation = 12.5% → **PLASTIC**

---

## Classification Logic

```python
if (max_stress > 1.1 × ξ₀) OR (max_deviation > 0.1):
    classification = 'PLASTIC'
else:
    classification = 'ELASTIC'
```

**A grain is ELASTIC if**:
- Maximum stress ≤ 1.1 × ξ₀ (hasn't yielded)
- **AND** maximum deviation ≤ 10% (follows elastic line closely)

**A grain is PLASTIC if**:
- Maximum stress > 1.1 × ξ₀ (has yielded), **OR**
- Maximum deviation > 10% (shows non-linear behavior)

---

## Physical Interpretation

### Elastic Grains

**Characteristics**:
- Follow **Hooke's law**: σ = E × ε
- Stress-strain curve is **linear**
- Haven't reached yield stress (ξ₀)
- No permanent deformation
- Reversible behavior

**Typical Properties**:
- High yield stress (ξ₀) relative to applied stress
- Low applied strain (below yield point)
- Simple stress-strain relationship

### Plastic Grains

**Characteristics**:
- **Deviate from Hooke's law**
- Stress-strain curve shows **hardening** (non-linear)
- Have reached or exceeded yield stress (ξ₀)
- Show permanent deformation
- Irreversible behavior

**Typical Properties**:
- Low yield stress (ξ₀) relative to applied stress
- High applied strain (above yield point)
- Complex stress-strain relationship with hardening

---

## Why Two Criteria?

### Criterion 1 (Yield Stress) - Direct Physical Check
- **Directly checks** if grain has yielded
- Based on material property (ξ₀)
- Clear physical meaning

### Criterion 2 (Deviation) - Behavior-Based Check
- **Catches** grains that show plastic behavior even if they don't exceed ξ₀
- Accounts for:
  - Microstructure effects (grain boundaries, neighbors)
  - Stress concentrations
  - Early yielding due to constraints
- More sensitive to actual behavior

**Together**, these criteria provide a robust classification that captures both:
1. **Material-based** yielding (Criterion 1)
2. **Behavior-based** plasticity (Criterion 2)

---

## Parameters

### `yield_threshold` (default: 0.1 = 10%)

**What it controls**:
- Maximum allowed deviation from elastic line
- Lower values = stricter (more grains classified as plastic)
- Higher values = more lenient (fewer grains classified as plastic)

**Typical values**:
- `0.05` (5%): Very strict - only clear plastic behavior
- `0.1` (10%): Default - balanced classification
- `0.15` (15%): Lenient - only obvious plastic behavior

**Recommendation**: Use default (0.1) unless you have specific requirements

---

## Examples

### Example 1: Clear Elastic Grain

**Properties**:
- E = 250 GPa
- ξ₀ = 300 MPa
- Applied strain: 0.0% to 0.4%

**Behavior**:
- Max stress = 180 MPa
- Elastic line: σ = 250 GPa × 0.004 = 1000 MPa (at 0.4% strain)
- Actual stress follows elastic line closely
- Deviation < 5%

**Classification**: **ELASTIC**
- ✅ max_stress (180 MPa) < 1.1 × ξ₀ (330 MPa)
- ✅ deviation < 10%

### Example 2: Clear Plastic Grain

**Properties**:
- E = 200 GPa
- ξ₀ = 150 MPa
- Applied strain: 0.0% to 0.4%

**Behavior**:
- Max stress = 250 MPa
- Stress exceeds elastic line significantly
- Shows hardening behavior
- Deviation > 15%

**Classification**: **PLASTIC**
- ✅ max_stress (250 MPa) > 1.1 × ξ₀ (165 MPa)
- ✅ deviation > 10%

### Example 3: Borderline Case

**Properties**:
- E = 200 GPa
- ξ₀ = 200 MPa
- Applied strain: 0.0% to 0.4%

**Behavior**:
- Max stress = 210 MPa (just above ξ₀)
- Stress slightly exceeds elastic line
- Deviation = 8%

**Classification**: **ELASTIC** (with default threshold)
- ❌ max_stress (210 MPa) < 1.1 × ξ₀ (220 MPa) - close but not exceeded
- ❌ deviation (8%) < 10% - within threshold

**Note**: With stricter threshold (5%), this might be classified as plastic

---

## Implementation Details

### Code Location
Function: `classify_grain_behavior()` in `analyze_test_per_grain_ss_curves.py`

### Algorithm Steps

1. **Load grain properties**: E, ξ₀ from props array
2. **Compute elastic estimate**: σ_elastic = E × ε for all strain values
3. **Calculate deviation**: |σ_actual - σ_elastic| / σ_elastic
4. **Check Criterion 1**: max(σ_actual) > 1.1 × ξ₀?
5. **Check Criterion 2**: max(deviation) > yield_threshold?
6. **Classify**: PLASTIC if either criterion is true, else ELASTIC

### Edge Cases

- **No data**: Returns 'unknown'
- **Invalid grain_id**: Returns 'unknown'
- **Zero elastic stress**: Uses small epsilon (1e-6) to avoid division by zero
- **Very small grains**: May have noisy data, but still classified

---

## Validation

### How to Verify Classification

1. **Visual Inspection**: Check stress-strain plots
   - Elastic grains: Linear curve, follows E×ε line
   - Plastic grains: Non-linear curve, exceeds E×ε line

2. **Property Correlation**: 
   - Low ξ₀ → More likely plastic
   - High ξ₀ → More likely elastic

3. **Spatial Patterns**:
   - Check microstructure map
   - Plastic grains may cluster in high-stress regions

### Expected Results

For typical microstructures at 0.4% strain:
- **Elastic grains**: 30-50% (grains with high ξ₀)
- **Plastic grains**: 50-70% (grains with low ξ₀ or high stress)
- **Unknown**: <5% (insufficient data)

---

## Use in Analysis

### What This Classification Enables

1. **Model Performance Analysis**:
   - Compare MAE/RMSE for plastic vs elastic grains
   - Identify if model struggles with plasticity

2. **Property-Behavior Correlation**:
   - Understand which properties lead to plasticity
   - Visualize spatial patterns

3. **Physics Validation**:
   - Verify model respects material physics
   - Check if plastic behavior is predicted correctly

### Key Questions Answered

- ✅ **Does the ML model predict plastic grains well?**
  - Compare error metrics for plastic vs elastic grains
  
- ✅ **What properties characterize plastic grains?**
  - Analyze property distributions (ξ₀, E, etc.)
  
- ✅ **Are there spatial patterns?**
  - Visualize behavior map in microstructure

---

## Summary

The classification uses **two complementary criteria**:
1. **Yield stress check**: Direct physical criterion
2. **Deviation check**: Behavior-based criterion

This ensures robust classification that captures both:
- Material-based yielding
- Behavior-based plasticity

The default threshold (10% deviation) provides a balanced classification suitable for most analyses.

---

**For questions or adjustments**: Modify `yield_threshold` parameter in `classify_grain_behavior()` function.




