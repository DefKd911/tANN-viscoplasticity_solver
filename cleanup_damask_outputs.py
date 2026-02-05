#!/usr/bin/env python3
"""
cleanup_damask_outputs.py

Clean up DAMASK output files from the parent directory.
Removes run_*.hdf5 and run_*.sta files that DAMASK creates.

Usage:
    python cleanup_damask_outputs.py [--dry-run]
"""

import glob
import os
import sys

def cleanup_damask_outputs(dry_run=False):
    """
    Remove DAMASK output files from the current directory.
    
    Args:
        dry_run: If True, only print what would be deleted without actually deleting
    """
    patterns = [
        "run_*.hdf5",
        "run_*.sta"
    ]
    
    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(glob.glob(pattern))
    
    # Filter to only include files (not directories)
    files_to_delete = [f for f in files_to_delete if os.path.isfile(f)]
    
    if not files_to_delete:
        print("[INFO] No DAMASK output files found in current directory")
        return 0
    
    print(f"[INFO] Found {len(files_to_delete)} DAMASK output files")
    
    if dry_run:
        print("\n[DRY RUN] Would delete the following files:")
        for f in files_to_delete:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.1f} MB)")
        
        total_size = sum(os.path.getsize(f) for f in files_to_delete) / (1024 * 1024)
        print(f"\n[DRY RUN] Total size: {total_size:.1f} MB")
        print("[DRY RUN] Run without --dry-run to actually delete these files")
        return len(files_to_delete)
    
    # Actually delete
    deleted = 0
    total_size = 0
    
    for f in files_to_delete:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            total_size += size
            deleted += 1
            print(f"[DELETED] {f} ({size / (1024 * 1024):.1f} MB)")
        except Exception as e:
            print(f"[ERROR] Could not delete {f}: {e}")
    
    print(f"\n[SUCCESS] Deleted {deleted}/{len(files_to_delete)} files")
    print(f"[SUCCESS] Freed {total_size / (1024 * 1024):.1f} MB")
    
    return deleted

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    
    print("="*60)
    print("DAMASK Output Cleanup Utility")
    print("="*60)
    
    if dry_run:
        print("[MODE] Dry run - files will NOT be deleted\n")
    else:
        print("[MODE] Live run - files WILL be deleted\n")
    
    deleted = cleanup_damask_outputs(dry_run=dry_run)
    
    print("="*60)








