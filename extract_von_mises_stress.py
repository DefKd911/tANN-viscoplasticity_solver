#!/usr/bin/env python3
"""
extract_von_mises_stress.py

Extract and visualize von Mises stress field from DAMASK HDF5 outputs.
This script reads DAMASK simulation results and computes von Mises stress
from stress tensor components, then visualizes the field as a 2D heatmap.

Usage:
    python extract_von_mises_stress.py --hdf5 simulation_results/hdf5_files/seed1005154883.hdf5 --increment increment_20
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional, Tuple, Dict, Any


def find_stress_data(h5_file: str, increment: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Find stress tensor data in DAMASK HDF5 file.
    
    Args:
        h5_file: Path to HDF5 file
        increment: Increment name (e.g., 'increment_20')
        
    Returns:
        Tuple of (stress_tensor, metadata)
    """
    with h5py.File(h5_file, "r") as f:
        print(f"[INFO] Examining file: {h5_file}")
        print(f"[INFO] Looking for increment: {increment}")
        
        if increment not in f:
            available_increments = [k for k in f.keys() if k.startswith('increment_')]
            raise ValueError(f"Increment {increment} not found. Available: {available_increments}")
        
        inc_data = f[increment]
        
        # Check geometry info
        if 'geometry' in inc_data:
            geom_data = inc_data['geometry']
            print(f"[INFO] Geometry data available: {list(geom_data.keys())}")
            
            # Check for stress in geometry
            for key in geom_data.keys():
                if 'stress' in key.lower() or 'cauchy' in key.lower():
                    print(f"[INFO] Found potential stress data: {key}")
                    stress_data = geom_data[key][:]
                    print(f"[INFO] Stress data shape: {stress_data.shape}")
                    return stress_data, {'source': 'geometry', 'key': key}
        
        # Check homogenization data
        if 'homogenization' in inc_data:
            homog_data = inc_data['homogenization']
            print(f"[INFO] Homogenization data available: {list(homog_data.keys())}")
            
            # Look for stress in homogenization entries
            for homog_key in homog_data.keys():
                homog_entry = homog_data[homog_key]
                if 'mechanical' in homog_entry:
                    mech_data = homog_entry['mechanical']
                    print(f"[INFO] Mechanical data in {homog_key}: {list(mech_data.keys())}")
                    
                    for mech_key in mech_data.keys():
                        if 'stress' in mech_key.lower() or 'cauchy' in mech_key.lower():
                            print(f"[INFO] Found stress data: {homog_key}/{mech_key}")
                            stress_data = mech_data[mech_key][:]
                            print(f"[INFO] Stress data shape: {stress_data.shape}")
                            return stress_data, {'source': 'homogenization', 'key': f"{homog_key}/{mech_key}"}
        
        # Check phase data
        if 'phase' in inc_data:
            phase_data = inc_data['phase']
            print(f"[INFO] Phase data available: {list(phase_data.keys())}")
            
            # Look for stress in phase entries
            for phase_key in phase_data.keys():
                phase_entry = phase_data[phase_key]
                if 'mechanical' in phase_entry:
                    mech_data = phase_entry['mechanical']
                    print(f"[INFO] Mechanical data in {phase_key}: {list(mech_data.keys())}")
                    
                    for mech_key in mech_data.keys():
                        if 'stress' in mech_key.lower() or 'cauchy' in mech_key.lower():
                            print(f"[INFO] Found stress data: {phase_key}/{mech_key}")
                            stress_data = mech_data[mech_key][:]
                            print(f"[INFO] Stress data shape: {stress_data.shape}")
                            return stress_data, {'source': 'phase', 'key': f"{phase_key}/{mech_key}"}
        
        # If no stress found, let's examine the full structure
        print(f"[WARNING] No stress data found. Full increment structure:")
        print_increment_structure(inc_data)
        
        raise ValueError("No stress tensor data found in the specified increment")


def print_increment_structure(group, indent=0):
    """Recursively print HDF5 group structure."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("  " * indent + f"[{key}] (Group)")
            print_increment_structure(item, indent+1)
        elif isinstance(item, h5py.Dataset):
            print("  " * indent + f"{key} (Dataset) shape={item.shape} dtype={item.dtype}")


def compute_von_mises_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stress from stress tensor.
    
    Args:
        stress_tensor: Stress tensor array of shape (N, 3, 3) or (N, 6) for Voigt notation
        
    Returns:
        von Mises stress array of shape (N,)
    """
    if stress_tensor.ndim == 3 and stress_tensor.shape[1:] == (3, 3):
        # Full tensor format (3x3)
        # Compute deviatoric stress
        trace = np.trace(stress_tensor, axis1=1, axis2=2)
        deviatoric = stress_tensor - trace[:, None, None] / 3.0 * np.eye(3)[None, :, :]
        
        # von Mises stress: sqrt(3/2 * s_ij * s_ij)
        von_mises = np.sqrt(1.5 * np.sum(deviatoric**2, axis=(1, 2)))
        
    elif stress_tensor.ndim == 2 and stress_tensor.shape[1] == 6:
        # Voigt notation (6 components: xx, yy, zz, xy, xz, yz)
        s11, s22, s33, s12, s13, s23 = stress_tensor.T
        
        # Compute von Mises stress from Voigt components
        von_mises = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 
                                   6.0 * (s12**2 + s13**2 + s23**2)))
    else:
        raise ValueError(f"Unsupported stress tensor shape: {stress_tensor.shape}")
    
    return von_mises


def reshape_to_2d(von_mises: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Reshape von Mises stress array to 2D field.
    
    Args:
        von_mises: 1D von Mises stress array
        target_shape: Target 2D shape (height, width). If None, assumes square grid.
        
    Returns:
        2D von Mises stress field
    """
    n_points = len(von_mises)
    
    if target_shape is None:
        # Assume square grid
        side_len = int(np.sqrt(n_points))
        if side_len * side_len != n_points:
            raise ValueError(f"Cannot reshape {n_points} points to square grid")
        target_shape = (side_len, side_len)
    
    if target_shape[0] * target_shape[1] != n_points:
        raise ValueError(f"Cannot reshape {n_points} points to {target_shape}")
    
    return von_mises.reshape(target_shape)


def visualize_von_mises_field(von_mises_field: np.ndarray, 
                            increment: str, 
                            output_file: str = "sigma_vm_field.png",
                            title: str = None) -> None:
    """
    Visualize von Mises stress field as 2D heatmap.
    
    Args:
        von_mises_field: 2D von Mises stress field
        increment: Increment name for title
        output_file: Output image file path
        title: Custom title (if None, auto-generated)
    """
    # Convert to MPa for better readability
    von_mises_mpa = von_mises_field / 1e6
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(von_mises_mpa, cmap="inferno", interpolation='nearest')
    
    if title is None:
        title = f"Von Mises Stress Field — {increment}"
    plt.title(title, fontsize=14)
    
    plt.xlabel("X direction", fontsize=12)
    plt.ylabel("Y direction", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label(r"$\sigma_{vM}$ [MPa]", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved visualization: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize von Mises stress from DAMASK HDF5 files")
    parser.add_argument("--hdf5", required=True, help="Path to DAMASK HDF5 file")
    parser.add_argument("--increment", default="increment_20", help="Increment to analyze (default: increment_20)")
    parser.add_argument("--output", default="sigma_vm_field.png", help="Output image file (default: sigma_vm_field.png)")
    parser.add_argument("--shape", help="Target 2D shape as 'height,width' (e.g., '64,64')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5):
        raise FileNotFoundError(f"HDF5 file not found: {args.hdf5}")
    
    try:
        # Extract stress data
        print(f"[INFO] Extracting stress data from {args.hdf5}")
        stress_tensor, metadata = find_stress_data(args.hdf5, args.increment)
        
        print(f"[INFO] Stress tensor shape: {stress_tensor.shape}")
        print(f"[INFO] Data source: {metadata['source']}/{metadata['key']}")
        
        # Compute von Mises stress
        print(f"[INFO] Computing von Mises stress...")
        von_mises = compute_von_mises_stress(stress_tensor)
        print(f"[INFO] von Mises stress range: {von_mises.min()/1e6:.2f} - {von_mises.max()/1e6:.2f} MPa")
        
        # Reshape to 2D
        target_shape = None
        if args.shape:
            try:
                target_shape = tuple(map(int, args.shape.split(',')))
            except ValueError:
                print(f"[WARNING] Invalid shape format: {args.shape}. Using auto-detection.")
        
        print(f"[INFO] Reshaping to 2D field...")
        von_mises_field = reshape_to_2d(von_mises, target_shape)
        print(f"[INFO] Field shape: {von_mises_field.shape}")
        
        # Visualize
        print(f"[INFO] Creating visualization...")
        visualize_von_mises_field(von_mises_field, args.increment, args.output)
        
        print(f"[SUCCESS] Analysis complete!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
