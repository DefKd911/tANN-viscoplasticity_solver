#!/usr/bin/env python3
"""
modify_material_files_for_stress_output.py

Modify DAMASK material YAML files to enable stress tensor output.
This script updates all material files to include 'stress_Cauchy' in the output list.

Usage:
    python modify_material_files_for_stress_output.py
    python modify_material_files_for_stress_output.py --backup  # Create backup files
"""

import os
import glob
import yaml
import argparse
import shutil
from pathlib import Path


def modify_material_file(file_path: str, backup: bool = False) -> bool:
    """
    Modify a single material YAML file to include stress_Cauchy output.
    
    Args:
        file_path: Path to material YAML file
        backup: Whether to create backup files
        
    Returns:
        True if modification was successful, False otherwise
    """
    try:
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            print(f"[BACKUP] Created backup: {backup_path}")
        
        # Load YAML file
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'phase' not in data:
            print(f"[SKIP] No phase data in {file_path}")
            return False
        
        modified = False
        
        # Modify each phase
        for phase_name, phase_data in data['phase'].items():
            if 'mechanical' in phase_data and 'plastic' in phase_data['mechanical']:
                plastic_data = phase_data['mechanical']['plastic']
                
                if 'output' in plastic_data:
                    # Check if stress_Cauchy is already in output
                    if 'stress_Cauchy' not in plastic_data['output']:
                        plastic_data['output'].append('stress_Cauchy')
                        modified = True
                        print(f"[MODIFY] Added stress_Cauchy to {phase_name} in {file_path}")
                    else:
                        print(f"[SKIP] stress_Cauchy already in {phase_name} output")
                else:
                    # Add output list with stress_Cauchy
                    plastic_data['output'] = ['xi', 'stress_Cauchy']
                    modified = True
                    print(f"[MODIFY] Added output list with stress_Cauchy to {phase_name} in {file_path}")
        
        # Save modified file
        if modified:
            with open(file_path, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
            print(f"[SUCCESS] Modified {file_path}")
            return True
        else:
            print(f"[SKIP] No modifications needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to modify {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Modify DAMASK material files to enable stress output")
    parser.add_argument("--backup", action="store_true", help="Create backup files before modification")
    parser.add_argument("--material-dir", default="material_yaml", help="Directory containing material files")
    parser.add_argument("--pattern", default="material_*.yaml", help="File pattern to match")
    
    args = parser.parse_args()
    
    # Find all material files
    material_files = glob.glob(os.path.join(args.material_dir, args.pattern))
    
    if not material_files:
        print(f"[ERROR] No material files found in {args.material_dir} matching {args.pattern}")
        return 1
    
    print(f"[INFO] Found {len(material_files)} material files")
    
    if args.backup:
        print(f"[INFO] Creating backup files...")
    
    # Modify each file
    modified_count = 0
    for file_path in sorted(material_files):
        if modify_material_file(file_path, args.backup):
            modified_count += 1
    
    print(f"\n[SUMMARY] Modified {modified_count} out of {len(material_files)} files")
    
    if modified_count > 0:
        print(f"\n[INFO] Next steps:")
        print(f"1. Re-run DAMASK simulations:")
        print(f"   python batch_run_damask.py")
        print(f"2. Extract von Mises stress:")
        print(f"   python extract_von_mises_stress.py --hdf5 <file> --increment <increment>")
    
    return 0


if __name__ == "__main__":
    exit(main())
