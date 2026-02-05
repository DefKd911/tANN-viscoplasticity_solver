# GT Overlays Analysis Report

## Overview
This document analyzes the GT overlay visualizations in `ML_EVAL/gt_overlays/` to verify they meet our requirements.

## Current Status

### ✅ What's Working Well

1. **7-Panel Layout**: The visualizations correctly show:
   - Panel 1: E (GPa)
   - Panel 2: nu (-)
   - Panel 3: xi0 (MPa)
   - Panel 4: h0 (GPa)
   - Panel 5: σvM (MPa) - Ground truth stress
   - Panel 6: σvM_el (MPa) - Elastic estimate
   - Panel 7: ΔσvM (MPa) - Deviation from elastic

2. **Stress Scaling**: Stress values are correctly converted from Pa to MPa using `sigma_scale=1e-6`.

3. **Property Units**: All properties are displayed in correct physical units:
   - E: GPa (converted from Pa)
   - nu: dimensionless
   - xi0: MPa (converted from Pa)
   - h0: GPa (converted from Pa)

4. **GB Overlay Alignment**: Auto-alignment is implemented using D4 transforms to match stress fields with labels.

### ⚠️ Issues Found

#### 1. **Incorrect Elastic Estimate Formula** (CRITICAL)

**Location**: `visualize_gt_from_hdf5.py`, line 558

**Current Code**:
```python
elastic_vm_mpa = (E * eps11 / (1.0 + np.clip(nu, 1e-9, 0.499999))) / 1e6
```

**Problem**: This formula divides by `(1+ν)`, which is incorrect for von Mises stress in uniaxial tension.

**Correct Formula**:
For uniaxial tension, the stress-strain relationship is:
- σ11 = E × ε11 (Hooke's law)
- Von Mises stress in uniaxial tension: σ_vM = σ11

Therefore: **σ_vM_el = E × ε11**

**Fix Required**:
```python
elastic_vm_mpa = (E * eps11) / 1e6
```

**Impact**: 
- The elastic estimate panels (Panel 6) are showing values that are too low (divided by ~1.3-1.5 for typical ν=0.3-0.5).
- The deviation panels (Panel 7) are showing larger deviations than they should, making it harder to identify truly plastic regions.

#### 2. **Missing `origin='lower'` and `interpolation='nearest'` in imshow**

**Location**: `visualize_gt_from_hdf5.py`, lines 447-449

**Current Code**:
```python
im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_)
```

**Issue**: Without `origin='lower'` and `interpolation='nearest'`, there could be slight misalignments between the stress maps and GB overlays, especially if the data was saved with different conventions.

**Recommended Fix**:
```python
im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_, origin='lower', interpolation='nearest')
```

**Impact**: Minor - the auto-alignment may compensate for this, but it's best practice for pixel-perfect alignment.

## Recommendations

### Immediate Actions

1. **Fix the elastic estimate formula** (Priority: HIGH)
   - Change line 558 in `visualize_gt_from_hdf5.py` to use `E * eps11` instead of `E * eps11 / (1+ν)`
   - Regenerate all GT overlay images

2. **Add `origin='lower'` and `interpolation='nearest'`** (Priority: MEDIUM)
   - Update `imshow` calls in `draw_panels` function
   - This ensures pixel-perfect alignment with GB overlays

### Verification Steps

After fixes, verify:
1. Elastic estimate values are reasonable (should match E × ε11)
2. Deviation panels show smaller values in elastic regions
3. GB overlays align perfectly with stress patterns
4. Stress values are in reasonable MPa range (0-1000 MPa typical)

## Physics Validation

### Expected Behavior

1. **Elastic Regions**: 
   - σvM ≈ σvM_el (E × ε11)
   - ΔσvM ≈ 0

2. **Plastic Regions**:
   - σvM > σvM_el (due to hardening)
   - ΔσvM > 0 (positive deviation)

3. **Property-Stress Correlation**:
   - High E regions → Higher stress (stiffer)
   - High xi0 regions → Higher yield stress
   - High h0 regions → Steeper hardening (more visible at higher strains)

### Current Status

With the incorrect formula, the elastic estimates are systematically too low, which makes:
- All regions appear more "plastic" than they actually are
- Harder to distinguish between elastic and plastic regions
- The deviation maps less informative

## Conclusion

The GT overlays are **mostly correct** but have one **critical issue** with the elastic estimate formula. Once fixed, the visualizations will provide accurate physical insights into the material behavior.

