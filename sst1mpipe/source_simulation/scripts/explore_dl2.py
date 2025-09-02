#!/usr/bin/env python3
"""
Simple script to explore DL2 file structure
"""

import h5py
import numpy as np
from pathlib import Path

def explore_hdf5_group(group, path="", max_depth=3, current_depth=0):
    """Recursively explore HDF5 group structure"""
    if current_depth >= max_depth:
        return
    
    indent = "  " * current_depth
    
    for key in group.keys():
        item = group[key]
        current_path = f"{path}/{key}" if path else key
        
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {current_path}")
            explore_hdf5_group(item, current_path, max_depth, current_depth + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}Dataset: {current_path} - shape: {item.shape}, dtype: {item.dtype}")
            if current_depth == 0:  # Only show first few values for top-level datasets
                try:
                    print(f"{indent}  First 5 values: {item[:5]}")
                except:
                    print(f"{indent}  Could not read values")

def explore_hdf5(filepath, max_depth=3):
    """Explore HDF5 file structure"""
    with h5py.File(filepath, 'r') as f:
        explore_hdf5_group(f, max_depth=max_depth, current_depth=0)

def find_event_data(filepath):
    """Find and display event data structure"""
    with h5py.File(filepath, 'r') as f:
        # Look for event data in common locations
        possible_paths = [
            'dl1/event',
            'dl2/event', 
            'dl1',
            'dl2',
            'events',
            'data'
        ]
        
        for path in possible_paths:
            if path in f:
                print(f"\nFound data at: {path}")
                group = f[path]
                print(f"Keys in {path}: {list(group.keys())}")
                
                # If it's a group, explore its contents
                if isinstance(group, h5py.Group):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            print(f"  Dataset: {key} - shape: {item.shape}, dtype: {item.dtype}")
                            try:
                                print(f"    First 3 values: {item[:3]}")
                            except:
                                print(f"    Could not read values")
                break
        else:
            print("Could not find event data in common locations")

def explore_telescope_data(filepath):
    """Explore telescope event data specifically"""
    with h5py.File(filepath, 'r') as f:
        if 'dl1/event/telescope' in f:
            print(f"\nExploring dl1/event/telescope:")
            telescope_group = f['dl1/event/telescope']
            print(f"Keys: {list(telescope_group.keys())}")
            
            for key in telescope_group.keys():
                item = telescope_group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  Dataset: {key} - shape: {item.shape}, dtype: {item.dtype}")
                    try:
                        print(f"    First 3 values: {item[:3]}")
                    except:
                        print(f"    Could not read values")
        
        if 'dl2/event/telescope' in f:
            print(f"\nExploring dl2/event/telescope:")
            telescope_group = f['dl2/event/telescope']
            print(f"Keys: {list(telescope_group.keys())}")
            
            for key in telescope_group.keys():
                item = telescope_group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  Dataset: {key} - shape: {item.shape}, dtype: {item.dtype}")
                    try:
                        print(f"    First 3 values: {item[:3]}")
                    except:
                        print(f"    Could not read values")

def explore_all_data(filepath):
    """Explore all data in the file to find the actual event data"""
    with h5py.File(filepath, 'r') as f:
        print(f"\nExploring all data in {filepath.name}:")
        print("=" * 60)
        
        def explore_recursive(group, path="", depth=0):
            if depth > 4:  # Limit depth
                return
            
            indent = "  " * depth
            for key in group.keys():
                item = group[key]
                current_path = f"{path}/{key}" if path else key
                
                if isinstance(item, h5py.Dataset):
                    print(f"{indent}Dataset: {current_path}")
                    print(f"{indent}  Shape: {item.shape}")
                    print(f"{indent}  Dtype: {item.dtype}")
                    
                    # Show column names if it's a structured array
                    if hasattr(item.dtype, 'names') and item.dtype.names:
                        print(f"{indent}  Columns: {item.dtype.names}")
                        # Show first few rows
                        try:
                            print(f"{indent}  First 2 rows:")
                            for i in range(min(2, len(item))):
                                print(f"{indent}    Row {i}: {item[i]}")
                        except:
                            print(f"{indent}    Could not read values")
                    else:
                        try:
                            print(f"{indent}  First 3 values: {item[:3]}")
                        except:
                            print(f"{indent}    Could not read values")
                    print()
                
                elif isinstance(item, h5py.Group):
                    print(f"{indent}Group: {current_path}")
                    explore_recursive(item, current_path, depth + 1)
        
        explore_recursive(f)

if __name__ == "__main__":
    # Test with one file
    test_file = Path("../dl2_gamma/gamma_point_50_300E3GeV_20_20deg_testing_dl1_dl2.h5")
    
    if test_file.exists():
        print(f"Exploring {test_file.name}:")
        print("=" * 50)
        explore_hdf5(test_file, max_depth=3)
        find_event_data(test_file)
        explore_telescope_data(test_file)
        explore_all_data(test_file)
    else:
        print(f"File not found: {test_file}")
