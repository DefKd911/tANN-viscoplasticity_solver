#!/usr/bin/env python3
"""
Quick script to check if HDF5 files contain stress_Cauchy data
"""
import h5py
import sys
from pathlib import Path

STRESS_DATASETS = ["stress_Cauchy", "sigma", "stress"]

def check_hdf5_for_stress(hdf5_path):
    """Check if HDF5 file contains stress_Cauchy data"""
    print(f"\n{'='*60}")
    print(f"Checking: {hdf5_path.name}")
    print('='*60)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check if we have any increments
        increments = [k for k in f.keys() if k.startswith('increment_')]
        if not increments:
            print("[X] NO INCREMENTS FOUND")
            return False
        
        print(f"[OK] Found {len(increments)} increments: {increments}")
        
        # Check first increment
        inc_key = increments[0]
        inc = f[inc_key]
        
        print(f"\nChecking {inc_key}:")
        print(f"  Groups: {list(inc.keys())}")
        
        has_stress = False
        
        def _check_group(label, mech_group):
            nonlocal has_stress
            mech_keys = list(mech_group.keys())
            print(f"    {label}: {mech_keys}")
            found = any(name in mech_group for name in STRESS_DATASETS)
            if not found and "output" in mech_group:
                out = mech_group["output"]
                out_keys = list(out.keys())
                print(f"      output/: {out_keys}")
                found = any(name in out for name in STRESS_DATASETS)
            if found:
                print(f"      [OK] FOUND stress data in {label}")
                has_stress = True
            else:
                print(f"      [X] NO stress data in {label}")
        
        # Check phase data
        if 'phase' in inc:
            phase_keys = list(inc['phase'].keys())
            print(f"\n  Phase groups: {phase_keys}")
            
            for pk in phase_keys[:2]:  # Check first 2 phases
                pg = inc['phase'][pk]
                if 'mechanical' in pg:
                    _check_group(f"{pk}/mechanical", pg['mechanical'])
                else:
                    print(f"    {pk}: No mechanical group")
        
        # Check homogenization data
        if 'homogenization' in inc:
            homog_keys = list(inc['homogenization'].keys())
            print(f"\n  Homogenization groups: {homog_keys}")
            
            for hk in homog_keys[:1]:  # Check first homog
                hg = inc['homogenization'][hk]
                if 'mechanical' in hg:
                    _check_group(f"{hk}/mechanical", hg['mechanical'])
                else:
                    print(f"    {hk}: No mechanical group")
        
        print(f"\n{'='*60}")
        if has_stress:
            print("[OK] RESULT: stress_Cauchy DATA FOUND - FILE IS USABLE")
        else:
            print("[X] RESULT: NO stress_Cauchy - FILE NEEDS REGENERATION")
        print('='*60)
        
        return has_stress

if __name__ == "__main__":
    # Check a few HDF5 files
    hdf5_dir = Path("simulation_results/hdf5_files")
    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))[:3]  # Check first 3
    
    if not hdf5_files:
        print("No HDF5 files found in simulation_results/hdf5_files/")
        sys.exit(1)
    
    results = []
    for hdf5_file in hdf5_files:
        has_stress = check_hdf5_for_stress(hdf5_file)
        results.append((hdf5_file.name, has_stress))
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for name, has_stress in results:
        status = "[OK] HAS STRESS" if has_stress else "[X] NO STRESS"
        print(f"{name:30s} {status}")
    
    total_with_stress = sum(1 for _, has in results if has)
    print(f"\nFiles with stress_Cauchy: {total_with_stress}/{len(results)}")
    
    if total_with_stress == 0:
        print("\n[!] ALL CHECKED FILES LACK STRESS DATA")
        print("    -> Need to regenerate simulations with updated material files")
    elif total_with_stress < len(results):
        print("\n[!] MIXED RESULTS - Some files have stress, some don't")
        print("    -> Older files need regeneration")
    else:
        print("\n[OK] ALL FILES HAVE STRESS DATA - READY FOR ANALYSIS")

