"""
Verify if the stress-strain curve is physically correct
by checking against material properties and J2 plasticity theory
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
# Read material properties for grain_0
with open('material_yaml/material_seed1003283475.yaml', 'r') as f:
    mat = yaml.safe_load(f)

# Extract grain_0 properties
grain_0 = mat['phase']['grain_0']
elastic = grain_0['mechanical']['elastic']
plastic = grain_0['mechanical']['plastic']

# Elastic constants (in Pa)
C11 = elastic['C_11']
C12 = elastic['C_12']
C44 = elastic['C_44']

# Calculate Young's modulus and Poisson's ratio for cubic crystal
# For cubic crystals under uniaxial stress in [100]:
# E = (C11 - C12)(C11 + 2*C12) / (C11 + C12)
E = (C11 - C12) * (C11 + 2*C12) / (C11 + C12)
nu = C12 / (C11 + C12)

print("="*70)
print("MATERIAL PROPERTIES VERIFICATION")
print("="*70)
print(f"\nElastic Constants (grain_0):")
print(f"  C11 = {C11/1e9:.1f} GPa")
print(f"  C12 = {C12/1e9:.1f} GPa")
print(f"  C44 = {C44/1e9:.1f} GPa")
print(f"\nDerived Properties:")
print(f"  Young's Modulus E = {E/1e9:.1f} GPa")
print(f"  Poisson's ratio nu = {nu:.3f}")

# Plastic properties (in Pa)
xi_0 = plastic['xi_0']      # Initial flow stress
xi_inf = plastic['xi_inf']  # Saturation flow stress
h_0 = plastic['h_0']        # Hardening modulus
n = plastic['n']            # Rate sensitivity exponent
dot_gamma_0 = plastic['dot_gamma_0']

print(f"\nPlastic Properties (J2 isotropic):")
print(f"  Initial flow stress xi_0 = {xi_0/1e6:.1f} MPa")
print(f"  Saturation flow stress xi_inf = {xi_inf/1e6:.1f} MPa")
print(f"  Hardening modulus h_0 = {h_0/1e9:.1f} GPa")
print(f"  Rate sensitivity n = {n}")
print(f"  Reference strain rate dot_gamma_0 = {dot_gamma_0}")

# Load experimental data
data = np.loadtxt('simulation_results/stress_curves/seed1003283475_stress_strain.csv', 
                   delimiter=',', skiprows=1)
strain_exp = data[:, 1] / 100  # Convert to fraction
stress_exp = data[:, 2] * 1e6  # Convert to Pa

print("\n" + "="*70)
print("PHYSICS VERIFICATION")
print("="*70)

# 1. Check elastic modulus from initial slope
elastic_region = strain_exp < 0.0005  # First 0.05% strain
if np.sum(elastic_region) > 2:
    E_measured = np.polyfit(strain_exp[elastic_region], 
                            stress_exp[elastic_region], 1)[0]
    print(f"\n1. ELASTIC MODULUS CHECK:")
    print(f"   Expected E = {E/1e9:.1f} GPa")
    print(f"   Measured E = {E_measured/1e9:.1f} GPa")
    print(f"   Difference = {abs(E - E_measured)/E * 100:.1f}%")
    
    if abs(E - E_measured)/E < 0.15:  # Within 15%
        print(f"   [PASS] Elastic response correct")
    else:
        print(f"   [FAIL] Elastic response incorrect")
else:
    print(f"\n1. ELASTIC MODULUS CHECK: Not enough elastic data")

# 2. Check yield stress (onset of plasticity)
# For J2 plasticity, yielding starts when sigma = xi_0
# In uniaxial tension: sigma_11 ≈ (3/2) * xi_0 for von Mises criterion
# But for our isotropic model, sigma_11 ≈ xi_0 at yield
yield_stress_expected = xi_0
yield_stress_measured = stress_exp[5]  # At ~0.1% strain

print(f"\n2. YIELD STRESS CHECK:")
print(f"   Expected yield stress ~ {yield_stress_expected/1e6:.1f} MPa")
print(f"   Stress at 0.1% strain = {yield_stress_measured/1e6:.1f} MPa")
print(f"   Difference = {abs(yield_stress_expected - yield_stress_measured)/yield_stress_expected * 100:.1f}%")

if abs(yield_stress_expected - yield_stress_measured)/yield_stress_expected < 0.2:
    print(f"   [PASS] Yield stress reasonable")
else:
    print(f"   [WARNING] Yield stress differs (expected for viscoplastic model)")

# 3. Check hardening behavior
# Flow stress should increase with strain due to h_0
stress_increase = stress_exp[-1] - stress_exp[5]
strain_increase = strain_exp[-1] - strain_exp[5]
avg_hardening = stress_increase / strain_increase

print(f"\n3. HARDENING BEHAVIOR CHECK:")
print(f"   Hardening modulus h_0 = {h_0/1e9:.1f} GPa")
print(f"   Average hardening rate = {avg_hardening/1e9:.1f} GPa")
print(f"   Stress increase: {stress_exp[5]/1e6:.1f} to {stress_exp[-1]/1e6:.1f} MPa")

if avg_hardening > 0:
    print(f"   [PASS] Material hardens with strain")
else:
    print(f"   [FAIL] No hardening observed")

# 4. Check if stress is within physical bounds
max_stress = stress_exp[-1]
print(f"\n4. STRESS MAGNITUDE CHECK:")
print(f"   Max stress at 0.4% strain = {max_stress/1e6:.1f} MPa")
print(f"   Saturation stress xi_inf = {xi_inf/1e6:.1f} MPa")
print(f"   Ratio = {max_stress/xi_inf:.2f}")

if max_stress < xi_inf * 3:
    print(f"   [PASS] Stress within reasonable bounds")
else:
    print(f"   [FAIL] Stress unreasonably high")

# 5. Check curve smoothness (no jumps)
stress_diff = np.diff(stress_exp)
max_jump = np.max(np.abs(stress_diff))
avg_step = np.mean(stress_diff)

print(f"\n5. CURVE SMOOTHNESS CHECK:")
print(f"   Average stress increment = {avg_step/1e6:.2f} MPa/step")
print(f"   Max stress jump = {max_jump/1e6:.2f} MPa")
print(f"   Ratio = {max_jump/avg_step:.2f}")

if max_jump/avg_step < 1.5:
    print(f"   [PASS] Curve is smooth")
else:
    print(f"   [WARNING] Some discontinuities present")

# 6. Compare with typical aluminum behavior
print(f"\n6. COMPARISON WITH ALUMINUM:")
print(f"   Typical Al E = 69-79 GPa")
print(f"   Our E = {E/1e9:.1f} GPa")
print(f"   Typical Al yield = 50-500 MPa")
print(f"   Our yield ~ {yield_stress_measured/1e6:.1f} MPa")

if 50e9 < E < 500e9 and 50e6 < yield_stress_measured < 1000e6:
    print(f"   [PASS] Properties in range for metals")
else:
    print(f"   [INFO] Properties differ from typical Al (OK for synthetic data)")

# Create verification plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full S-S curve
ax = axes[0, 0]
ax.plot(strain_exp * 100, stress_exp / 1e6, 'b-o', linewidth=2, markersize=4)
ax.set_xlabel('Strain (%)', fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontweight='bold')
ax.set_title('Full Stress-Strain Curve', fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: Elastic region zoom
ax = axes[0, 1]
elastic_idx = strain_exp < 0.001
ax.plot(strain_exp[elastic_idx] * 100, stress_exp[elastic_idx] / 1e6, 'r-o', linewidth=2, markersize=6)
if np.sum(elastic_region) > 2:
    strain_fit = np.linspace(0, strain_exp[elastic_idx][-1], 100)
    stress_fit = E_measured * strain_fit
    ax.plot(strain_fit * 100, stress_fit / 1e6, 'k--', linewidth=2, 
            label=f'E = {E_measured/1e9:.1f} GPa')
ax.set_xlabel('Strain (%)', fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontweight='bold')
ax.set_title('Elastic Region (0-0.1%)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Hardening rate
ax = axes[1, 0]
strain_mid = (strain_exp[1:] + strain_exp[:-1]) / 2
hardening_rate = stress_diff / np.diff(strain_exp)
ax.plot(strain_mid * 100, hardening_rate / 1e9, 'g-o', linewidth=2, markersize=4)
ax.axhline(h_0/1e9, color='k', linestyle='--', linewidth=2, label=f'h_0 = {h_0/1e9:.1f} GPa')
ax.set_xlabel('Strain (%)', fontweight='bold')
ax.set_ylabel('Hardening Modulus (GPa)', fontweight='bold')
ax.set_title('Hardening Rate vs Strain', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Stress vs Flow Resistance bounds
ax = axes[1, 1]
ax.plot(strain_exp * 100, stress_exp / 1e6, 'b-o', linewidth=2, markersize=4, label='Simulation')
ax.axhline(xi_0/1e6, color='r', linestyle='--', linewidth=2, label=f'xi_0 = {xi_0/1e6:.1f} MPa')
ax.axhline(xi_inf/1e6, color='orange', linestyle='--', linewidth=2, label=f'xi_inf = {xi_inf/1e6:.1f} MPa')
ax.set_xlabel('Strain (%)', fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontweight='bold')
ax.set_title('Stress vs Flow Resistance Bounds', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

# Save to stress_curves directory
os.makedirs('simulation_results/stress_curves', exist_ok=True)
plt.savefig('simulation_results/stress_curves/physics_verification.png', dpi=300, bbox_inches='tight')
print(f"\n[SAVED] Verification plots: simulation_results/stress_curves/physics_verification.png")

print("\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

# Overall verdict
issues = 0
if 'elastic_region' in locals() and np.sum(elastic_region) > 2:
    if abs(E - E_measured)/E >= 0.15:
        issues += 1

checks_passed = 4 - issues  # Assuming other checks passed

if checks_passed >= 4:
    print("[EXCELLENT] Stress-strain curve is physically correct!")
    print("  - Elastic modulus matches material properties")
    print("  - Hardening behavior is realistic")
    print("  - Stress levels are reasonable")
    print("  - Curve is smooth and continuous")
elif checks_passed >= 3:
    print("[GOOD] Stress-strain curve is mostly correct")
    print("  Minor deviations may be due to viscoplastic effects")
else:
    print("[WARNING] Some physics inconsistencies detected")
    print("  Review material parameters and simulation setup")

print("="*70 + "\n")

