#!/usr/bin/env python3
import sys
import h5py

def walk(name, obj, depth=0, max_depth=8):
    if depth > max_depth:
        return
    indent = '  ' * depth
    if isinstance(obj, h5py.Dataset):
        shape = obj.shape
        print(f"{indent}[D] {name}  shape={shape}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}[G] {name}")
        for k in obj.keys():
            try:
                walk(f"{name}/{k}", obj[k], depth+1, max_depth)
            except Exception as e:
                print(f"{indent}  (error entering {name}/{k}: {e})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python h5_inspect.py <file.hdf5> [max_depth]")
        sys.exit(1)
    path = sys.argv[1]
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with h5py.File(path, 'r') as h5:
        walk('', h5, depth=0, max_depth=max_depth)

if __name__ == '__main__':
    main()






