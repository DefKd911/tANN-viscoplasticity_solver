#!/usr/bin/env python3
"""
Analyze and visualize why different grains have different hardening slopes.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the summary data
with open('per_grain_analysis/seed105447566_summary.json', 'r') as f:
    summary = json.load(f)

grains = summary['grain_classifications']

# Extract data
grain_ids = []
E_values = []
xi0_values = []
h0_values = []
classifications = []

for gid, data in grains.items():
    grain_ids.append(int(gid))
    E_values.append(data['E'] / 1e9)  # GPa
    xi0_values.append(data['xi0'] / 1e6)  # MPa
    h0_values.append(data['h0'] / 1e9)  # GPa
    classifications.append(data['classification'])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Hardening Modulus (h0) vs Grain ID
ax1 = axes[0, 0]
colors = ['red' if c == 'plastic' else 'blue' for c in classifications]
bars = ax1.bar(grain_ids, h0_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Grain ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Hardening Modulus h0 (GPa)', fontsize=12, fontweight='bold')
ax1.set_title('Hardening Modulus (h0) for Each Grain\n(h0 = Hardening Slope)', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(['Plastic', 'Elastic'], loc='upper right')
for i, (gid, h0) in enumerate(zip(grain_ids, h0_values)):
    ax1.text(gid, h0 + 1, f'{h0:.1f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Expected Hardening Slopes Comparison
ax2 = axes[0, 1]
# Simulate stress-strain curves showing different slopes
strain = np.linspace(0, 0.4, 100)  # 0 to 0.4% strain

# Select a few representative grains
representative_grains = [3, 5, 6, 8]  # Different h0 values
for gid in representative_grains:
    idx = grain_ids.index(gid)
    E = E_values[idx]
    xi0 = xi0_values[idx]
    h0 = h0_values[idx]
    
    # Calculate expected yield strain
    yield_strain = (xi0 / (E * 1000)) * 100  # Convert to %
    
    # Elastic region
    elastic_strain = strain[strain <= yield_strain]
    elastic_stress = (E * 1000) * (elastic_strain / 100)  # Convert to MPa
    
    # Hardening region
    if yield_strain < strain.max():
        plastic_strain = strain[strain > yield_strain]
        plastic_stress = xi0 + (h0 * 1000) * ((plastic_strain - yield_strain) / 100)
        
        # Plot
        ax2.plot(elastic_strain, elastic_stress, 'g-', linewidth=2, alpha=0.7)
        ax2.plot(plastic_strain, plastic_stress, 'r-', linewidth=2, 
                label=f'Grain {gid} (h0={h0:.1f} GPa)')
        ax2.plot([yield_strain], [xi0], 'yo', markersize=8, markeredgecolor='black', markeredgewidth=1.5)

ax2.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
ax2.set_title('Different Hardening Slopes (h0) Create Different Curves', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9)

# Plot 3: h0 vs E scatter
ax3 = axes[1, 0]
plastic_mask = [c == 'plastic' for c in classifications]
elastic_mask = [c == 'elastic' for c in classifications]

ax3.scatter([E_values[i] for i in range(len(E_values)) if plastic_mask[i]], 
           [h0_values[i] for i in range(len(h0_values)) if plastic_mask[i]], 
           c='red', s=100, alpha=0.7, label='Plastic', edgecolors='black')
ax3.scatter([E_values[i] for i in range(len(E_values)) if elastic_mask[i]], 
           [h0_values[i] for i in range(len(h0_values)) if elastic_mask[i]], 
           c='blue', s=100, alpha=0.7, label='Elastic', edgecolors='black')

# Annotate grains
for i, gid in enumerate(grain_ids):
    ax3.annotate(f'G{gid}', (E_values[i], h0_values[i]), 
                fontsize=8, ha='center', va='bottom')

ax3.set_xlabel('Young\'s Modulus E (GPa)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Hardening Modulus h0 (GPa)', fontsize=12, fontweight='bold')
ax3.set_title('Material Properties: E vs h0\n(No correlation - independent properties)', 
              fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Hardening slope comparison table
ax4 = axes[1, 1]
ax4.axis('off')

# Create table data
table_data = []
table_data.append(['Grain', 'E (GPa)', 'xi0 (MPa)', 'h0 (GPa)', 'Elastic Slope', 'Hardening Slope', 'Type'])
for i, gid in enumerate(sorted(grain_ids)):
    E = E_values[grain_ids.index(gid)]
    xi0 = xi0_values[grain_ids.index(gid)]
    h0 = h0_values[grain_ids.index(gid)]
    ctype = classifications[grain_ids.index(gid)]
    table_data.append([
        f'{gid}',
        f'{E:.0f}',
        f'{xi0:.0f}',
        f'{h0:.1f}',
        f'{E:.0f} GPa',
        f'{h0:.1f} GPa',
        ctype.capitalize()
    ])

table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center',
                 colWidths=[0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code rows
for i in range(1, len(table_data)):
    grain_idx = grain_ids.index(int(table_data[i][0]))
    if classifications[grain_idx] == 'plastic':
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor('#ffcccc')
    else:
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor('#ccccff')

ax4.set_title('Material Properties Summary\nKey: Hardening Slope = h0 (different for each grain!)', 
              fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('per_grain_analysis/hardening_slopes_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: per_grain_analysis/hardening_slopes_analysis.png")

# Print summary
print("\n" + "="*70)
print("HARDENING SLOPES ANALYSIS")
print("="*70)
print("\nKey Finding: Each grain has a DIFFERENT h0 (hardening modulus)")
print("This creates DIFFERENT hardening slopes in the stress-strain curves!\n")

print("Hardening Modulus (h0) Values:")
for gid in sorted(grain_ids):
    idx = grain_ids.index(gid)
    print(f"  Grain {gid}: h0 = {h0_values[idx]:.1f} GPa  "
          f"(E={E_values[idx]:.0f} GPa, xi0={xi0_values[idx]:.0f} MPa, "
          f"Type={classifications[idx]})")

print("\n" + "="*70)
print("EXPLANATION:")
print("="*70)
print("""
1. ELASTIC REGION (Green Line):
   - Slope = E (Young's Modulus)
   - Different for each grain (50-300 GPa range)
   - Formula: σ = E × ε

2. HARDENING REGION (Red Line):
   - Slope = h0 (Hardening Modulus)  
   - VERY different for each grain (0.7-45 GPa range)
   - Formula: σ = xi0 + h0 × ε_plastic
   - This is why you see different slopes!

3. WHY SEPARATE REGIONS?
   - Different physics (elastic vs plastic)
   - Different slopes (E vs h0)
   - Clear transition at yield point

4. ELASTIC GRAINS:
   - Have high E (stiff)
   - May have low h0 or yield very late
   - Stay close to elastic line throughout
""")

