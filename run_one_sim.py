#!/usr/bin/env python3
import os
import sys
from glob import glob
from pathlib import Path

import batch_run_damask as brd

# Get seed from command line or environment
if len(sys.argv) > 1:
    seed = sys.argv[1] if sys.argv[1].startswith("seed") else f"seed{sys.argv[1]}"
else:
    seed = os.environ.get("SEED", "seed1003283475")

def find_pair(seed: str):
	geom = os.path.join(brd.GEOM_DIR, f"run_{seed}.vti")
	mat = os.path.join(brd.MATERIAL_DIR, f"material_{seed}.yaml")
	if not os.path.exists(geom):
		raise FileNotFoundError(f"Geometry not found: {geom}")
	if not os.path.exists(mat):
		raise FileNotFoundError(f"Material not found: {mat}")
	return geom, mat

if __name__ == "__main__":
	brd.setup_output_directories()
	if not brd.check_damask_installation():
		raise SystemExit(1)
	geom, mat = find_pair(seed)
	print(f"[RUN_ONE] Running single simulation for {seed}")
	ok = brd.run_damask_simulation(geom, mat, seed)
	print(f"[RUN_ONE] Result: {'SUCCESS' if ok else 'FAILED'}")
