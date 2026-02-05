#!/usr/bin/env python3
"""
analyze_damask_stress_data.py

Comprehensive analysis of DAMASK stress-related data.
This script provides multiple analysis modes:

1. Current data analysis (xi field visualization)
2. Batch processing of multiple files
3. Comparison between different increments
4. Preparation for ML training data

Usage:
    python analyze_damask_stress_data.py --mode current --hdf5 damask_outputs/run_seed1012767171_load_material_seed1012767171_numerics.hdf5
    python analyze_damask_stress_data.py --mode batch --input-dir damask_outputs
    python analyze_damask_stress_data.py --mode compare --hdf5 damask_outputs/run_seed1012767171_load_material_seed1012767171_numerics.hdf5
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class DAMASKAnalyzer:
    """Comprehensive DAMASK data analyzer."""
    
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        self.file_info = self._get_file_info()
    
    def _get_file_info(self) -> Dict:
        """Get basic file information."""
        with h5py.File(self.h5_file, "r") as f:
            increment_keys = [k for k in f.keys() if k.startswith('increment_')]
            increment_keys.sort(key=lambda x: int(x.split('_')[1]))
            
            return {
                'increments': increment_keys,
                'n_increments': len(increment_keys),
                'file_size': os.path.getsize(self.h5_file) / (1024*1024)  # MB
            }
    
    def extract_xi_field(self, increment: str) -> Tuple[np.ndarray, Dict]:
        """Extract xi field from specified increment."""
        with h5py.File(self.h5_file, "r") as f:
            if increment not in f:
                raise ValueError(f"Increment {increment} not found")
            
            inc_data = f[increment]
            xi_data = []
            grain_info = []
            
            if 'phase' in inc_data:
                phase_data = inc_data['phase']
                grain_keys = sorted([k for k in phase_data.keys() if k.startswith('grain_')])
                
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
            
            if not xi_data:
                raise ValueError("No xi data found")
            
            xi_combined = np.concatenate(xi_data)
            
            metadata = {
                'n_grains': len(grain_keys),
                'total_points': len(xi_combined),
                'grain_info': grain_info,
                'xi_range': (xi_combined.min(), xi_combined.max())
            }
            
            return xi_combined, metadata
    
    def visualize_xi_field(self, xi_field: np.ndarray, increment: str, 
                          output_file: str = None, title: str = None) -> None:
        """Visualize xi field as 2D heatmap."""
        # Reshape to 2D (assuming square grid)
        n_points = len(xi_field)
        side_len = int(np.sqrt(n_points))
        if side_len * side_len != n_points:
            raise ValueError(f"Cannot reshape {n_points} points to square grid")
        
        xi_field_2d = xi_field.reshape((side_len, side_len))
        xi_mpa = xi_field_2d / 1e6  # Convert to MPa
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(xi_mpa, cmap="plasma", interpolation='nearest')
        
        if title is None:
            title = f"Flow Resistance (ξ) Field — {increment}"
        plt.title(title, fontsize=14)
        
        plt.xlabel("X direction", fontsize=12)
        plt.ylabel("Y direction", fontsize=12)
        
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label(r"$\xi$ [MPa]", fontsize=12)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Saved visualization: {output_file}")
        
        plt.show()
    
    def compare_increments(self, increments: List[str] = None) -> None:
        """Compare xi fields across different increments."""
        if increments is None:
            increments = self.file_info['increments']
        
        n_increments = len(increments)
        fig, axes = plt.subplots(1, n_increments, figsize=(5*n_increments, 5))
        
        if n_increments == 1:
            axes = [axes]
        
        for i, increment in enumerate(increments):
            try:
                xi_field, metadata = self.extract_xi_field(increment)
                
                # Reshape to 2D
                n_points = len(xi_field)
                side_len = int(np.sqrt(n_points))
                xi_field_2d = xi_field.reshape((side_len, side_len))
                xi_mpa = xi_field_2d / 1e6
                
                im = axes[i].imshow(xi_mpa, cmap="plasma", interpolation='nearest')
                axes[i].set_title(f"{increment}\nξ range: {metadata['xi_range'][0]/1e6:.1f} - {metadata['xi_range'][1]/1e6:.1f} MPa")
                axes[i].set_xlabel("X")
                axes[i].set_ylabel("Y")
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{increment}:\n{e}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{increment} (Error)")
        
        plt.tight_layout()
        plt.show()
    
    def batch_analyze_files(self, input_dir: str, pattern: str = "*.hdf5") -> Dict:
        """Analyze multiple HDF5 files in batch."""
        h5_files = glob.glob(os.path.join(input_dir, pattern))
        
        if not h5_files:
            raise ValueError(f"No HDF5 files found in {input_dir} matching {pattern}")
        
        results = {}
        
        for h5_file in sorted(h5_files):
            try:
                analyzer = DAMASKAnalyzer(h5_file)
                file_name = os.path.basename(h5_file)
                
                # Get basic info
                results[file_name] = {
                    'file_info': analyzer.file_info,
                    'status': 'success'
                }
                
                # Try to extract data from last increment
                if analyzer.file_info['increments']:
                    last_increment = analyzer.file_info['increments'][-1]
                    xi_field, metadata = analyzer.extract_xi_field(last_increment)
                    results[file_name]['xi_metadata'] = metadata
                    results[file_name]['last_increment'] = last_increment
                
                print(f"[SUCCESS] Analyzed {file_name}")
                
            except Exception as e:
                results[file_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"[ERROR] Failed to analyze {file_name}: {e}")
        
        return results
    
    def prepare_ml_data(self, increment: str, output_file: str = None) -> np.ndarray:
        """Prepare xi field data for ML training."""
        xi_field, metadata = self.extract_xi_field(increment)
        
        # Reshape to 2D
        n_points = len(xi_field)
        side_len = int(np.sqrt(n_points))
        xi_field_2d = xi_field.reshape((side_len, side_len))
        
        # Normalize to [0, 1] range
        xi_min, xi_max = xi_field_2d.min(), xi_field_2d.max()
        xi_normalized = (xi_field_2d - xi_min) / (xi_max - xi_min)
        
        if output_file:
            np.save(output_file, xi_normalized)
            print(f"[SUCCESS] Saved ML data: {output_file}")
        
        return xi_normalized


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DAMASK stress data analysis")
    parser.add_argument("--mode", choices=["current", "batch", "compare", "ml"], 
                       default="current", help="Analysis mode")
    parser.add_argument("--hdf5", help="Path to HDF5 file (for current/compare/ml modes)")
    parser.add_argument("--increment", default="increment_20", help="Increment to analyze")
    parser.add_argument("--input-dir", default="damask_outputs", help="Input directory for batch mode")
    parser.add_argument("--pattern", default="*.hdf5", help="File pattern for batch mode")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.mode == "current":
        if not args.hdf5:
            print("[ERROR] --hdf5 required for current mode")
            return 1
        
        if not os.path.exists(args.hdf5):
            print(f"[ERROR] HDF5 file not found: {args.hdf5}")
            return 1
        
        analyzer = DAMASKAnalyzer(args.hdf5)
        print(f"[INFO] File: {args.hdf5}")
        print(f"[INFO] Available increments: {analyzer.file_info['increments']}")
        print(f"[INFO] File size: {analyzer.file_info['file_size']:.1f} MB")
        
        try:
            xi_field, metadata = analyzer.extract_xi_field(args.increment)
            print(f"[INFO] Xi field shape: {xi_field.shape}")
            print(f"[INFO] Xi range: {metadata['xi_range'][0]/1e6:.2f} - {metadata['xi_range'][1]/1e6:.2f} MPa")
            
            analyzer.visualize_xi_field(xi_field, args.increment, args.output)
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return 1
    
    elif args.mode == "batch":
        if not os.path.exists(args.input_dir):
            print(f"[ERROR] Input directory not found: {args.input_dir}")
            return 1
        
        analyzer = DAMASKAnalyzer("dummy")  # We'll create new instances
        results = analyzer.batch_analyze_files(args.input_dir, args.pattern)
        
        print(f"\n[SUMMARY] Batch analysis complete:")
        print(f"Total files: {len(results)}")
        print(f"Successful: {sum(1 for r in results.values() if r['status'] == 'success')}")
        print(f"Failed: {sum(1 for r in results.values() if r['status'] == 'error')}")
        
        # Show summary statistics
        successful_results = [r for r in results.values() if r['status'] == 'success']
        if successful_results:
            print(f"\n[STATISTICS]")
            print(f"Average file size: {np.mean([r['file_info']['file_size'] for r in successful_results]):.1f} MB")
            print(f"Average increments: {np.mean([r['file_info']['n_increments'] for r in successful_results]):.1f}")
    
    elif args.mode == "compare":
        if not args.hdf5:
            print("[ERROR] --hdf5 required for compare mode")
            return 1
        
        analyzer = DAMASKAnalyzer(args.hdf5)
        analyzer.compare_increments()
    
    elif args.mode == "ml":
        if not args.hdf5:
            print("[ERROR] --hdf5 required for ml mode")
            return 1
        
        analyzer = DAMASKAnalyzer(args.hdf5)
        ml_data = analyzer.prepare_ml_data(args.increment, args.output)
        print(f"[INFO] ML data shape: {ml_data.shape}")
        print(f"[INFO] ML data range: {ml_data.min():.3f} - {ml_data.max():.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())

