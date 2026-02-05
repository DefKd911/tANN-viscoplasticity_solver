#!/usr/bin/env python3
"""
Fix DAMASK material files to ensure stress_Cauchy output works properly.

In DAMASK 3.x with 'pass' homogenization:
- Phase-level outputs (like stress_Cauchy) should be at homogenization level
- Phase plastic outputs should be phase-specific quantities (like xi)
"""

import yaml
from pathlib import Path
import shutil

def fix_material_file(yaml_path, backup=True):
    """
    Ensure homogenization has stress_Cauchy output
    """
    if backup and not Path(f"{yaml_path}.backup2").exists():
        shutil.copy2(yaml_path, f"{yaml_path}.backup2")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    modified = False
    
    # Fix homogenization outputs
    if 'homogenization' in data:
        for hkey, hval in data['homogenization'].items():
            if 'mechanical' in hval:
                if 'output' not in hval['mechanical']:
                    hval['mechanical']['output'] = []
                    modified = True
                
                # Ensure stress_Cauchy is in homogenization output
                if 'stress_Cauchy' not in hval['mechanical']['output']:
                    hval['mechanical']['output'].append('stress_Cauchy')
                    modified = True
                    print(f"  [+] Added stress_Cauchy to {hkey}/mechanical/output")
    
    # Phase outputs - keep xi and stress_Cauchy for now
    # (DAMASK will ignore stress_Cauchy at phase level if at homog level)
    if 'phase' in data:
        for pkey, pval in data['phase'].items():
            if 'mechanical' in pval and 'plastic' in pval['mechanical']:
                plastic = pval['mechanical']['plastic']
                if 'output' in plastic:
                    # Ensure both xi and stress_Cauchy are there
                    if 'xi' not in plastic['output']:
                        plastic['output'].insert(0, 'xi')
                        modified = True
                    if 'stress_Cauchy' not in plastic['output']:
                        plastic['output'].append('stress_Cauchy')
                        modified = True
    
    if modified:
        # Write back with proper formatting
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    return modified

def main():
    material_dir = Path("material_yaml")
    yaml_files = sorted(material_dir.glob("material_seed*.yaml"))
    
    # Test on a few files first
    test_files = yaml_files[:5]
    
    print("="*70)
    print("FIXING MATERIAL FILES FOR STRESS OUTPUT")
    print("="*70)
    print(f"\nTesting on {len(test_files)} files first...\n")
    
    for yaml_file in test_files:
        print(f"Processing: {yaml_file.name}")
        modified = fix_material_file(yaml_file, backup=True)
        if modified:
            print(f"  [OK] Updated")
        else:
            print(f"  [SKIP] Already correct")
    
    print("\n" + "="*70)
    print("Test complete! Check one of these files manually.")
    print("If OK, run with all 1000 files.")
    print("="*70)

if __name__ == "__main__":
    main()








