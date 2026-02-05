#!/usr/bin/env python3
"""
Batch generate stress-strain curves for all HDF5 files
Saves curves to simulation_results/stress_curves/
"""
import glob
import subprocess
import os

# Find all HDF5 files
hdf5_files = sorted(glob.glob('simulation_results/hdf5_files/seed*.hdf5'))

print(f"Found {len(hdf5_files)} HDF5 files")
print("Generating stress-strain curves...\n")

successful = 0
failed = 0

for i, hdf5_file in enumerate(hdf5_files, 1):
    seed_name = os.path.basename(hdf5_file).replace('.hdf5', '')
    
    # Check if curve already exists
    output_file = f'simulation_results/stress_curves/{seed_name}_stress_strain.png'
    if os.path.exists(output_file):
        print(f"[{i}/{len(hdf5_files)}] SKIP {seed_name} (already exists)")
        successful += 1
        continue
    
    try:
        # Run plot script
        result = subprocess.run(
            ['python', 'plot_ss_curve_from_F_P.py', hdf5_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[{i}/{len(hdf5_files)}] SUCCESS {seed_name}")
        successful += 1
    except subprocess.CalledProcessError as e:
        print(f"[{i}/{len(hdf5_files)}] FAILED {seed_name}: {e.stderr[:100]}")
        failed += 1

print(f"\n{'='*60}")
print(f"COMPLETE!")
print(f"{'='*60}")
print(f"Successful: {successful}/{len(hdf5_files)}")
print(f"Failed: {failed}/{len(hdf5_files)}")
print(f"Output directory: simulation_results/stress_curves/")
print(f"{'='*60}")








