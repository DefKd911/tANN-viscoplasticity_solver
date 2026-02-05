import h5py

h5_file = "simulation_results/hdf5_files/seed108835770.hdf5"

def print_tree(g, indent=0):
    for key in g.keys():
        item = g[key]
        if isinstance(item, h5py.Group):
            print("  " * indent + f"[{key}] (Group)")
            print_tree(item, indent+1)
        elif isinstance(item, h5py.Dataset):
            print("  " * indent + f"{key} (Dataset) shape={item.shape} dtype={item.dtype}")

with h5py.File(h5_file, "r") as f:
    print("Top-level keys:", list(f.keys()))
    
    # Check for increment data
    increment_keys = [k for k in f.keys() if k.startswith('increment_')]
    print(f"\nAvailable increments: {increment_keys}")
    
    # Also check for any other potential data structures
    print(f"\n--- Full file structure ---")
    print_tree(f)
    
    if increment_keys:
        # Examine the last increment
        last_increment = max(increment_keys, key=lambda x: int(x.split('_')[1]))
        print(f"\n--- {last_increment} ---")
        print_tree(f[last_increment])
        
        # Check phase data structure
        if 'phase' in f[last_increment]:
            print(f"\n--- Phase data in {last_increment} ---")
            phase_data = f[last_increment]['phase']
            print(f"Phase keys: {list(phase_data.keys())}")
            
            # Check first phase for mechanical data
            first_phase = list(phase_data.keys())[0]
            print(f"\n--- {first_phase} mechanical data ---")
            if 'mechanical' in phase_data[first_phase]:
                print_tree(phase_data[first_phase]['mechanical'])
        
        # Check homogenization data for stress tensors
        if 'homogenization' in f[last_increment]:
            print(f"\n--- Homogenization data in {last_increment} ---")
            homog_data = f[last_increment]['homogenization']
            print(f"Homogenization keys: {list(homog_data.keys())}")
            
            # Check first homogenization entry for mechanical data
            first_homog = list(homog_data.keys())[0]
            print(f"\n--- {first_homog} mechanical data ---")
            if 'mechanical' in homog_data[first_homog]:
                print_tree(homog_data[first_homog]['mechanical'])
