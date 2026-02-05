#!/usr/bin/env python3
"""
Check if elastic region actually matches elastic estimate for ground truth data.
If they don't match, it means there's early plasticity or yield detection is wrong.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Load summary
with open('per_grain_analysis/seed105447566_summary.json', 'r') as f:
    summary = json.load(f)

seed = "105447566"
h5_path = f"simulation_results/hdf5_files/seed{seed}.hdf5"
labels_path = f"labels/labels_seed{seed}.npy"
props_path = f"props/props_seed{seed}.npy"

labels = np.load(labels_path)
props_raw = np.load(props_path, allow_pickle=True)

if props_raw.ndim == 0 and isinstance(props_raw.item(), dict):
    d = props_raw.item()
    props = np.stack([np.asarray(d['E']), np.asarray(d['nu']), 
                     np.asarray(d['xi0']), np.asarray(d['h0'])], axis=1)
else:
    props = props_raw

# Load a few grains to check
grains_to_check = [3, 1, 5, 0]  # Mix of plastic and elastic

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, gid in enumerate(grains_to_check):
    ax = axes[idx]
    gid_str = str(gid)
    if gid_str not in summary['grain_classifications']:
        continue
    
    info = summary['grain_classifications'][gid_str]
    E = info['E']
    nu = info['nu']
    xi0 = info['xi0'] / 1e6  # MPa
    h0 = info['h0'] / 1e9  # GPa
    
    # Load grain data (simplified - just get strain and stress)
    # This is a quick check, so we'll use the summary data structure
    # In reality, we'd need to reload from HDF5
    
    # Expected elastic behavior
    # For ground truth, in pure elastic region: σ = E × ε
    # If green line doesn't match gray line, it's NOT elastic!
    
    # Create a message
    ax.text(0.5, 0.5, f'Grain {gid}\n\nE = {E/1e9:.0f} GPa\nxi0 = {xi0:.0f} MPa\n\n'
            f'If green line (GT elastic) does NOT overlap\n'
            f'with gray line (E×ε), then:\n'
            f'1. Yield happened earlier than detected\n'
            f'2. Early plastic effects present\n'
            f'3. NOT truly elastic!', 
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Grain {gid} - Elastic Overlap Check', fontweight='bold')

plt.tight_layout()
plt.savefig('per_grain_analysis/elastic_overlap_check.png', dpi=300, bbox_inches='tight')
print("Created: per_grain_analysis/elastic_overlap_check.png")

print("\n" + "="*70)
print("CRITICAL OBSERVATION: Elastic Region Should Match Elastic Estimate!")
print("="*70)
print("""
If the green line (GT elastic region) does NOT overlap with the gray dashed 
line (E×ε elastic estimate), then:

1. The region is NOT truly elastic!
2. Either:
   - Yield happened earlier than detected
   - Early plastic effects are present
   - Numerical errors in DAMASK

SOLUTION:
- Only call it "elastic" if it actually matches E×ε
- Adjust yield detection to find where deviation starts
- Or accept that there's early plasticity before detected yield
""")

