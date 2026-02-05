#!/usr/bin/env python3
"""
extract_and_visualize_stress.py

Extract and visualize stress-related fields from DAMASK HDF5 outputs.
Since stress tensor data is not available in current simulations, this script:
1. Extracts available data (xi - flow resistance)
2. Provides visualization of xi field as a proxy for stress
3. Shows how to modify DAMASK configuration for stress output

Usage:
    python extract_and_visualize_stress.py --hdf5 simulation_results/hdf5_files/seed1005154883.hdf5 --increment increment_20
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional, Tuple, Dict, Any


def extract_xi_field(h5_file: str, increment: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract xi (flow resistance) field from DAMASK HDF5 file.
    
    Args:
        h5_file: Path to HDF5 file
        increment: Increment name (e.g., 'increment_20')
        
    Returns:
        Tuple of (xi_field, metadata)
    """
    with h5py.File(h5_file, "r") as f:
        print(f"[INFO] Extracting xi field from {h5_file}")
        print(f"[INFO] Increment: {increment}")
        
        if increment not in f:
            available_increments = [k for k in f.keys() if k.startswith('increment_')]
            raise ValueError(f"Increment {increment} not found. Available: {available_increments}")
        
        inc_data = f[increment]
        
        # Extract xi data from all grains
        xi_data = []
        grain_info = []
        
        if 'phase' in inc_data:
            phase_data = inc_data['phase']
            grain_keys = sorted([k for k in phase_data.keys() if k.startswith('grain_')])
            
            print(f"[INFO] Found {len(grain_keys)} grains: {grain_keys}")
            
            for grain_key in grain_keys:
                grain_data = phase_data[grain_key]
                if 'mechanical' in grain_data and 'xi' in grain_data['mechanical']:
                    xi_grain = grain_data['mechanical']['xi'][:]
                    xi_data.append(xi_grain)
                    grain_info.append({
                        'grain_id': grain_key,
                        'n_points': len(xi_grain),
                        'xi_range': (xi_grain.min(), xi_grain.max())
                    })
                    print(f"[INFO] {grain_key}: {len(xi_grain)} points, xi range: {xi_grain.min()/1e6:.2f} - {xi_grain.max()/1e6:.2f} MPa")
        
        if not xi_data:
            raise ValueError("No xi data found in phase mechanical data")
        
        # Combine all xi data
        xi_combined = np.concatenate(xi_data)
        
        metadata = {
            'n_grains': len(grain_keys),
            'total_points': len(xi_combined),
            'grain_info': grain_info,
            'xi_range': (xi_combined.min(), xi_combined.max())
        }
        
        print(f"[INFO] Combined xi field: {len(xi_combined)} points")
        print(f"[INFO] Overall xi range: {xi_combined.min()/1e6:.2f} - {xi_combined.max()/1e6:.2f} MPa")
        
        return xi_combined, metadata


def reshape_to_2d(xi_field: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Reshape xi field to 2D.
    
    Args:
        xi_field: 1D xi field array
        target_shape: Target 2D shape (height, width). If None, assumes square grid.
        
    Returns:
        2D xi field
    """
    n_points = len(xi_field)
    
    if target_shape is None:
        # Assume square grid
        side_len = int(np.sqrt(n_points))
        if side_len * side_len != n_points:
            raise ValueError(f"Cannot reshape {n_points} points to square grid")
        target_shape = (side_len, side_len)
    
    if target_shape[0] * target_shape[1] != n_points:
        raise ValueError(f"Cannot reshape {n_points} points to {target_shape}")
    
    return xi_field.reshape(target_shape)


def visualize_xi_field(xi_field: np.ndarray, 
                      increment: str, 
                      output_file: str = "xi_field.png",
                      title: str = None) -> None:
    """
    Visualize xi field as 2D heatmap.
    
    Args:
        xi_field: 2D xi field
        increment: Increment name for title
        output_file: Output image file path
        title: Custom title (if None, auto-generated)
    """
    # Convert to MPa for better readability
    xi_mpa = xi_field / 1e6
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(xi_mpa, cmap="plasma", interpolation='nearest')
    
    if title is None:
        title = f"Flow Resistance (ξ) Field — {increment}"
    plt.title(title, fontsize=14)
    
    plt.xlabel("X direction", fontsize=12)
    plt.ylabel("Y direction", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label(r"$\xi$ [MPa]", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved visualization: {output_file}")
    plt.show()


def create_stress_output_config():
    """
    Create example DAMASK configuration files with stress output enabled.
    """
    print("\n" + "="*60)
    print("CONFIGURATION FOR STRESS OUTPUT")
    print("="*60)
    
    print("\nTo enable stress tensor output in DAMASK, modify your material YAML files:")
    print("Add 'stress_Cauchy' to the output list in each phase definition:")
    
    example_config = """
# Example phase configuration with stress output:
phase:
  grain_0:
    lattice: cF
    mechanical:
      elastic:
        type: Hooke
        C_11: 200e9
        C_12: 100e9
        C_44: 50e9
      plastic:
        type: isotropic
        output: [xi, stress_Cauchy]  # <-- Add stress_Cauchy here
        xi_0: 100e6
        xi_inf: 200e6
        h_0: 1e9
        # ... other parameters
"""
    
    print(example_config)
    
    print("\nAfter modifying the material files, re-run DAMASK simulations:")
    print("python batch_run_damask.py")
    
    print("\nThen use the full stress extraction script:")
    print("python extract_von_mises_stress.py --hdf5 <file> --increment <increment>")


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize xi field from DAMASK HDF5 files")
    parser.add_argument("--hdf5", required=True, help="Path to DAMASK HDF5 file")
    parser.add_argument("--increment", default="increment_20", help="Increment to analyze (default: increment_20)")
    parser.add_argument("--output", default="xi_field.png", help="Output image file (default: xi_field.png)")
    parser.add_argument("--shape", help="Target 2D shape as 'height,width' (e.g., '64,64')")
    parser.add_argument("--show-config", action="store_true", help="Show configuration for stress output")
    
    args = parser.parse_args()
    
    if args.show_config:
        create_stress_output_config()
        return 0
    
    if not os.path.exists(args.hdf5):
        raise FileNotFoundError(f"HDF5 file not found: {args.hdf5}")
    
    try:
        # Extract xi data
        print(f"[INFO] Extracting xi field from {args.hdf5}")
        xi_field, metadata = extract_xi_field(args.hdf5, args.increment)
        
        # Reshape to 2D
        target_shape = None
        if args.shape:
            try:
                target_shape = tuple(map(int, args.shape.split(',')))
            except ValueError:
                print(f"[WARNING] Invalid shape format: {args.shape}. Using auto-detection.")
        
        print(f"[INFO] Reshaping to 2D field...")
        xi_field_2d = reshape_to_2d(xi_field, target_shape)
        print(f"[INFO] Field shape: {xi_field_2d.shape}")
        
        # Visualize
        print(f"[INFO] Creating visualization...")
        visualize_xi_field(xi_field_2d, args.increment, args.output)
        
        print(f"[SUCCESS] Analysis complete!")
        print(f"[INFO] Note: This shows flow resistance (ξ) as a proxy for stress.")
        print(f"[INFO] For actual von Mises stress, modify DAMASK configuration and re-run simulations.")
        
        # Show configuration info
        create_stress_output_config()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
