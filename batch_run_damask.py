#!/usr/bin/env python3
"""
batch_run_damask.py

Automate running DAMASK simulations for all generated geometry and material files.
For each pair of .vti (geometry) and .yaml (material) files, runs DAMASK simulation
and saves the output as .hdf5 files.

Usage:
    python batch_run_damask.py
    python batch_run_damask.py --max 200
    python batch_run_damask.py --all
    python batch_run_damask.py --max 100 --parallel
    # Test data only (separate folder):
    python batch_run_damask.py --seeds-file test_seeds_to_simulate.txt --hdf5-dir simulation_results/test_hdf5_files
"""

import os
import glob
import subprocess
import shutil
import argparse
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Configuration
GEOM_DIR = "geom_vti"
MATERIAL_DIR = "material_yaml_fixed"
OUTPUT_DIR = "simulation_results"
HDF5_DIR = "hdf5_files"
STA_DIR = "sta_files"
LOG_DIR = "simulation_logs"
LOAD_YAML = "load.yaml"
NUMERICS_YAML = "numerics.yaml"
DAMASK_CMD = "DAMASK_grid"  # Adjust if DAMASK command is different
MAX_WORKERS = min(4, mp.cpu_count())  # Limit parallel workers for stability

def setup_output_directories():
    """Create structured output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, HDF5_DIR), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, STA_DIR), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, LOG_DIR), exist_ok=True)
    print(f"[INFO] Created output directories:")
    print(f"  - {OUTPUT_DIR}/")
    print(f"  - {OUTPUT_DIR}/{HDF5_DIR}/")
    print(f"  - {OUTPUT_DIR}/{STA_DIR}/")
    print(f"  - {OUTPUT_DIR}/{LOG_DIR}/")

def check_damask_installation():
    """Check if DAMASK is properly installed and accessible."""
    try:
        # DAMASK_grid doesn't support --version, so we'll try --help instead
        result = subprocess.run([DAMASK_CMD, "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[INFO] DAMASK found and working")
            return True
        else:
            # Even if --help fails, if we can execute the command, it exists
            print(f"[INFO] DAMASK command found (help returned code {result.returncode})")
            return True
    except FileNotFoundError:
        print(f"[ERROR] DAMASK command '{DAMASK_CMD}' not found in PATH")
        print("[SOLUTION] Please install DAMASK or update DAMASK_CMD variable with correct path")
        return False
    except subprocess.TimeoutExpired:
        print(f"[ERROR] DAMASK command timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking DAMASK: {e}")
        return False

def regenerate_problematic_material_files(failed_seeds):
    """Regenerate material files for seeds that failed due to material model errors."""
    if not failed_seeds:
        return
    
    print(f"[INFO] Regenerating material files for {len(failed_seeds)} failed seeds...")
    
    for seed in failed_seeds:
        try:
            # Find the corresponding props file
            props_file = os.path.join("props", f"props_{seed}.npy")
            if not os.path.exists(props_file):
                print(f"[WARNING] Props file not found for {seed}")
                continue
            
            # Regenerate material file
            material_file = os.path.join(MATERIAL_DIR, f"material_{seed}.yaml")
            cmd = ["python3", "export_material.py", "--props", props_file, "--out", material_file]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[SUCCESS] Regenerated material file for {seed}")
            else:
                print(f"[ERROR] Failed to regenerate material file for {seed}: {result.stderr}")
                
        except Exception as e:
            print(f"[ERROR] Exception regenerating material for {seed}: {e}")

def cleanup_remaining_outputs():
    """Clean up any remaining output files in the main directory and move them to proper locations."""
    print(f"[INFO] Cleaning up remaining output files...")
    
    # Find all HDF5 and STA files in the main directory
    main_dir_patterns = [
        "run_*_*.hdf5",
        "run_*_*.sta", 
        "*_load_material_numerics.hdf5",
        "*_load_material_numerics.sta"
    ]
    
    moved_count = 0
    for pattern in main_dir_patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file) and not file.startswith(OUTPUT_DIR):
                try:
                    if file.endswith('.hdf5'):
                        # Extract seed from filename
                        base_name = os.path.splitext(file)[0]
                        if 'run_seed' in base_name:
                            seed_name = base_name.split('run_seed')[1].split('_')[0]
                            dest_path = os.path.join(OUTPUT_DIR, HDF5_DIR, f"seed{seed_name}.hdf5")
                        else:
                            # Generic name
                            dest_path = os.path.join(OUTPUT_DIR, HDF5_DIR, os.path.basename(file))
                        shutil.move(file, dest_path)
                        print(f"[CLEANUP] Moved HDF5: {file} -> {dest_path}")
                        moved_count += 1
                    elif file.endswith('.sta'):
                        # Extract seed from filename
                        base_name = os.path.splitext(file)[0]
                        if 'run_seed' in base_name:
                            seed_name = base_name.split('run_seed')[1].split('_')[0]
                            dest_path = os.path.join(OUTPUT_DIR, STA_DIR, f"seed{seed_name}.sta")
                        else:
                            # Generic name
                            dest_path = os.path.join(OUTPUT_DIR, STA_DIR, os.path.basename(file))
                        shutil.move(file, dest_path)
                        print(f"[CLEANUP] Moved STA: {file} -> {dest_path}")
                        moved_count += 1
                except Exception as e:
                    print(f"[WARNING] Could not move {file}: {e}")
    
    if moved_count > 0:
        print(f"[INFO] Cleaned up {moved_count} remaining output files")
    else:
        print(f"[INFO] No remaining output files to clean up")

# Setup output directories will be called in main()

def run_single_simulation(args):
    """Wrapper function for parallel processing."""
    geom_file, material_file, output_name, out_hdf5_dir = args
    return run_damask_simulation(geom_file, material_file, output_name, out_hdf5_dir=out_hdf5_dir)

def run_damask_simulation(geom_file, material_file, output_name, material_errors_list=None, out_hdf5_dir=None):
    """
    Run DAMASK simulation for a single geometry/material pair.
    
    Args:
        geom_file: Path to .vti geometry file
        material_file: Path to .yaml material file  
        output_name: Base name for output files (without extension)
        material_errors_list: List to track material model errors (optional)
        out_hdf5_dir: If set, save HDF5 here; else use OUTPUT_DIR/HDF5_DIR
    """
    hdf5_dest_dir = out_hdf5_dir if out_hdf5_dir else os.path.join(OUTPUT_DIR, HDF5_DIR)
    # Clean up any potential leftover files first
    geom_base = os.path.splitext(os.path.basename(geom_file))[0]
    leftover_patterns = [
        f"{geom_base}_*.hdf5",
        f"{geom_base}_*.sta",
        f"run_*_load_material_numerics.hdf5",
        f"run_*_load_material_numerics.sta"
    ]
    
    for pattern in leftover_patterns:
        for leftover_file in glob.glob(pattern):
            try:
                os.remove(leftover_file)
            except:
                pass
    
    try:
        # DAMASK command: DAMASK_grid --geom geometry.vti --material material.yaml --load load.yaml --numerics numerics.yaml
        cmd = [
            DAMASK_CMD,
            "--geom", geom_file,
            "--material", material_file,
            "--load", LOAD_YAML,
            "--numerics", NUMERICS_YAML
        ]
        
        print(f"[DAMASK] Running simulation for {output_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # DAMASK typically creates output files with pattern: run_*_load_material_numerics.hdf5
        # We need to find and move the generated files to our output directory
        # Extract base name from geom_file (e.g., run_seed123456.vti -> run_seed123456)
        geom_base = os.path.splitext(os.path.basename(geom_file))[0]
        
        # Look for various possible output patterns
        hdf5_patterns = [
            f"{geom_base}_load_material_numerics.hdf5",
            f"{geom_base}_*.hdf5",
            "run_*_load_material_numerics.hdf5",
            "*.hdf5"
        ]
        
        hdf5_files = []
        for pattern in hdf5_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                hdf5_files.extend(found_files)
                break  # Use the first pattern that finds files
        
        # Remove duplicates and filter out files that might be in subdirectories
        hdf5_files = list(set([f for f in hdf5_files if os.path.isfile(f) and not f.startswith(OUTPUT_DIR)]))
        
        # Prioritize non-restart files (they contain increments); move restart files to logs
        primary_hdf5 = [f for f in hdf5_files if 'restart' not in f.lower()]
        restart_hdf5 = [f for f in hdf5_files if 'restart' in f.lower()]
        
        if hdf5_files:
            # Move primary (non-restart) HDF5 to target directory
            for hdf5_file in primary_hdf5:
                dest_path = os.path.join(hdf5_dest_dir, f"{output_name}.hdf5")
                try:
                    shutil.move(hdf5_file, dest_path)
                    print(f"[SUCCESS] HDF5: {hdf5_file} -> {dest_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to move {hdf5_file}: {e}")
            
            # Move restart files to simulation_logs (don't overwrite primary)
            for restart_file in restart_hdf5:
                dest_path = os.path.join(OUTPUT_DIR, LOG_DIR, os.path.basename(restart_file))
                try:
                    shutil.move(restart_file, dest_path)
                    print(f"[SUCCESS] Restart HDF5: {restart_file} -> {dest_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to move {restart_file}: {e}")
        else:
            print(f"[WARNING] No HDF5 output found for {output_name}")
            
        # Also move any .sta files if they exist
        sta_patterns = [
            f"{geom_base}_load_material_numerics.sta",
            f"{geom_base}_*.sta",
            "run_*_load_material_numerics.sta",
            "*.sta"
        ]
        
        sta_files = []
        for pattern in sta_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                sta_files.extend(found_files)
                break
        
        sta_files = list(set([f for f in sta_files if os.path.isfile(f) and not f.startswith(OUTPUT_DIR)]))
        
        for sta_file in sta_files:
            dest_path = os.path.join(OUTPUT_DIR, STA_DIR, f"{output_name}.sta")
            try:
                shutil.move(sta_file, dest_path)
                print(f"[SUCCESS] STA: {sta_file} -> {dest_path}")
            except Exception as e:
                print(f"[ERROR] Failed to move {sta_file}: {e}")
        
        return True  # Simulation completed successfully
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] DAMASK simulation failed for {output_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        
        # Check if it's a convergence error (cutback exceeded)
        if "cutbacks exceeded" in str(e.stderr):
            print(f"[INFO] Convergence failure - this is common with complex microstructures")
            print(f"[INFO] Consider adjusting numerics.yaml parameters or material properties")
        
        # Check if it's an HDF5 file error
        elif "HDF5" in str(e.stderr) or "unable to open file" in str(e.stderr):
            print(f"[INFO] HDF5 file error - this might be due to leftover files or permissions")
            print(f"[INFO] Try running cleanup_damask_files.py first")
        
        # Check if it's a material model error
        elif "inversion error" in str(e.stderr) or "crystallite responds elastically" in str(e.stderr):
            print(f"[INFO] Material model error - tangent calculation failed")
            print(f"[INFO] This often indicates extreme material properties or numerical issues")
            print(f"[INFO] Consider regenerating material files with more conservative parameters")
            if material_errors_list is not None:
                material_errors_list.append(output_name)
        
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error for {output_name}: {e}")
        return False

def main():
    """Main function to run batch DAMASK simulations."""
    
    # Setup structured output directories
    setup_output_directories()
    
    # Check if DAMASK is installed
    if not check_damask_installation():
        return
    
    # Check if required files exist
    if not os.path.exists(LOAD_YAML):
        print(f"[ERROR] Required file not found: {LOAD_YAML}")
        print("Please ensure load.yaml exists in the current directory.")
        return
    
    if not os.path.exists(NUMERICS_YAML):
        print(f"[ERROR] Required file not found: {NUMERICS_YAML}")
        print("Please ensure numerics.yaml exists in the current directory.")
        return
    
    # Find all geometry files
    geom_files = sorted(glob.glob(os.path.join(GEOM_DIR, "run_*.vti")))
    print(f"[INFO] Found {len(geom_files)} geometry files")
    
    # Find all material files
    material_files = sorted(glob.glob(os.path.join(MATERIAL_DIR, "material_*.yaml")))
    print(f"[INFO] Found {len(material_files)} material files")
    
    if len(geom_files) != len(material_files):
        print(f"[WARNING] Mismatch: {len(geom_files)} geometry files vs {len(material_files)} material files")
    
    # Parse CLI args (optional; if not set, prompt interactively)
    parser = argparse.ArgumentParser(description="Run DAMASK simulations and dump HDF5 files.")
    parser.add_argument("--max", type=int, default=None, help="Max number of simulations to run (skips already-done)")
    parser.add_argument("--all", action="store_true", help="Run all geometries (same as --max with total count)")
    parser.add_argument("--seeds-file", type=str, default=None, help="Run only seeds listed in file (one seed per line, e.g. for test set)")
    parser.add_argument("--hdf5-dir", type=str, default=None, help="Save HDF5 to this folder (e.g. simulation_results/test_hdf5_files for separate test data)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing (faster)")
    args = parser.parse_args()

    seeds_filter = None
    if args.seeds_file:
        if not os.path.isfile(args.seeds_file):
            print(f"[ERROR] Seeds file not found: {args.seeds_file}")
            return
        with open(args.seeds_file, "r") as f:
            seeds_filter = set(line.strip() for line in f if line.strip())
        print(f"[INFO] --seeds-file: will run only {len(seeds_filter)} seeds from {args.seeds_file} (skipping existing HDF5)")

    # Optional separate folder for HDF5 (e.g. test data)
    effective_hdf5_dir = args.hdf5_dir if args.hdf5_dir else os.path.join(OUTPUT_DIR, HDF5_DIR)
    if args.hdf5_dir:
        os.makedirs(effective_hdf5_dir, exist_ok=True)
        print(f"[INFO] --hdf5-dir: saving HDF5 to {effective_hdf5_dir}")
    
    if seeds_filter is not None:
        # Run only seeds from file; no max limit, no prompt
        max_simulations = len(geom_files)
    elif args.all:
        max_simulations = len(geom_files)
        print(f"[INFO] --all: will run up to {max_simulations} simulations (skipping existing HDF5)")
    elif args.max is not None:
        max_simulations = min(args.max, len(geom_files))
        print(f"[INFO] --max {args.max}: will run up to {max_simulations} simulations (skipping existing HDF5)")
    else:
        # Interactive prompt
        print(f"\n[INFO] How many simulations to run?")
        print(f"  - For baseline model: recommend 50")
        print(f"  - For full model: use {len(geom_files)} (all)")
        try:
            num_sims_input = input("Number of simulations (default=50, enter 'all' for all): ").strip()
            if num_sims_input.lower() == 'all':
                max_simulations = len(geom_files)
            elif num_sims_input == "":
                max_simulations = 50
            else:
                max_simulations = int(num_sims_input)
        except Exception:
            max_simulations = 50
        print(f"[INFO] Will run up to {max_simulations} simulations")
    
    if args.parallel:
        use_parallel = True
        print(f"[INFO] --parallel: using {MAX_WORKERS} workers")
    else:
        use_parallel = False
        if args.max is None and args.all is False:
            print(f"\n[INFO] Available processing modes:")
            print(f"  1. Sequential (slower but more stable)")
            print(f"  2. Parallel (faster, uses {MAX_WORKERS} workers)")
            try:
                choice = input("Choose processing mode (1 or 2, default=1): ").strip()
                use_parallel = choice == "2"
            except Exception:
                pass
    
    if use_parallel:
        print(f"[INFO] Using parallel processing with {MAX_WORKERS} workers")
    else:
        print(f"[INFO] Using sequential processing")
    
    # Prepare simulation tasks
    simulation_tasks = []
    for geom_file in geom_files:
        # Extract seed from geometry filename (e.g., run_seed123456.vti -> seed123456)
        base_name = os.path.splitext(os.path.basename(geom_file))[0]  # run_seed123456
        seed_name = base_name.replace("run_", "")  # seed123456

        # If --seeds-file: only include seeds from that file
        if seeds_filter is not None and seed_name not in seeds_filter:
            continue
        # Else: stop if we've reached the maximum number of simulations
        if seeds_filter is None and len(simulation_tasks) >= max_simulations:
            break
            
        # Find matching material file
        material_file = os.path.join(MATERIAL_DIR, f"material_{seed_name}.yaml")
        
        if not os.path.exists(material_file):
            print(f"[SKIP] No matching material file for {geom_file}")
            continue
        
        # Check if output already exists (in the chosen HDF5 dir)
        output_file = os.path.join(effective_hdf5_dir, f"{seed_name}.hdf5")
        if os.path.exists(output_file):
            print(f"[SKIP] Output already exists for {seed_name}")
            continue
        
        simulation_tasks.append((geom_file, material_file, seed_name, effective_hdf5_dir))
    
    target_str = str(len(seeds_filter)) if seeds_filter else str(max_simulations)
    print(f"[INFO] Prepared {len(simulation_tasks)} simulation tasks (target: {target_str})")
    
    # Process simulations
    processed = 0
    successful = 0
    failed = 0
    material_errors = []  # Track seeds with material model errors
    start_time = time.time()
    
    if use_parallel and len(simulation_tasks) > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_single_simulation, task): task for task in simulation_tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                seed_name = task[2]
                # task = (geom_file, material_file, seed_name, effective_hdf5_dir)
                
                try:
                    result = future.result()
                    processed += 1
                    
                    if result:
                        successful += 1
                    else:
                        failed += 1
                    
                    # Progress update
                    if processed % 10 == 0 or processed == len(simulation_tasks):
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"[PROGRESS] Processed {processed}/{len(simulation_tasks)} simulations (Success: {successful}, Failed: {failed}) - Rate: {rate:.1f} sim/s")
                        
                except Exception as e:
                    print(f"[ERROR] Exception in parallel processing for {seed_name}: {e}")
                    failed += 1
                    processed += 1
    else:
        # Sequential processing
        for task in simulation_tasks:
            geom_file, material_file, seed_name, out_hdf5_dir = task
            result = run_damask_simulation(geom_file, material_file, seed_name, material_errors, out_hdf5_dir=out_hdf5_dir)
            processed += 1
            
            if result:
                successful += 1
            else:
                failed += 1
            
            # Progress update every 10 simulations
            if processed % 10 == 0 or processed == len(simulation_tasks):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[PROGRESS] Processed {processed}/{len(simulation_tasks)} simulations (Success: {successful}, Failed: {failed}) - Rate: {rate:.1f} sim/s")
    
    # Clean up any remaining output files in the main directory
    cleanup_remaining_outputs()
    
    # Final cleanup: remove any leftover run_* files in parent directory
    print(f"\n[INFO] Final cleanup: removing leftover DAMASK output files from parent directory...")
    cleanup_count = 0
    for pattern in ["run_*.hdf5", "run_*.sta"]:
        leftover_files = glob.glob(pattern)
        for leftover in leftover_files:
            if os.path.isfile(leftover):
                try:
                    os.remove(leftover)
                    cleanup_count += 1
                    print(f"[CLEANUP] Deleted: {leftover}")
                except Exception as e:
                    print(f"[WARNING] Could not delete {leftover}: {e}")
    
    if cleanup_count > 0:
        print(f"[INFO] Removed {cleanup_count} leftover files from parent directory")
    else:
        print(f"[INFO] No leftover files found in parent directory")
    
    # Regenerate material files for seeds with material model errors
    if material_errors:
        print(f"\n[INFO] Found {len(material_errors)} simulations with material model errors")
        regenerate_problematic_material_files(material_errors)
    
    # Final statistics
    total_time = time.time() - start_time
    avg_time_per_sim = total_time / processed if processed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Batch DAMASK simulation finished!")
    print(f"{'='*60}")
    print(f"Total processed: {processed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/processed*100):.1f}%" if processed > 0 else "Success rate: 0%")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per simulation: {avg_time_per_sim:.1f} seconds")
    print(f"Processing mode: {'Parallel' if use_parallel else 'Sequential'}")
    if use_parallel:
        print(f"Parallel workers: {MAX_WORKERS}")
    print(f"\nOutput files saved in:")
    print(f"  - HDF5 files: {OUTPUT_DIR}/{HDF5_DIR}/")
    print(f"  - STA files:  {OUTPUT_DIR}/{STA_DIR}/")
    print(f"  - Logs:       {OUTPUT_DIR}/{LOG_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
