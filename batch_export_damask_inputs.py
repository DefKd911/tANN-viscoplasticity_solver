import os
import glob
import subprocess

LABELS_DIR = "labels"
PROPS_DIR = "props"
GEOM_SCRIPT = "export_geom.py"
MATERIAL_SCRIPT = "export_material.py"
GEOM_OUT_DIR = "geom_vti"
MATERIAL_OUT_DIR = "material_yaml"

os.makedirs(GEOM_OUT_DIR, exist_ok=True)
os.makedirs(MATERIAL_OUT_DIR, exist_ok=True)

print(f"[DEBUG] Current working directory: {os.getcwd()}")

label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "labels_*.npy")))

for label_path in label_files:
    base = os.path.splitext(os.path.basename(label_path))[0].replace("labels_", "")
    prop_path = os.path.join(PROPS_DIR, f"props_{base}.npy")
    print(f"[DEBUG] Label file: '{label_path}'")
    print(f"[DEBUG] Expected property file: '{prop_path}'")
    print(f"[DEBUG] File exists: {os.path.exists(prop_path)}")
    if not os.path.exists(prop_path):
        print(f"[SKIP] No matching property file for {label_path}")
        continue
    vti_out = os.path.join(GEOM_OUT_DIR, f"run_{base}.vti")
    yaml_out = os.path.join(MATERIAL_OUT_DIR, f"material_{base}.yaml")

    # Call export_geom.py
    subprocess.run([
        "python", GEOM_SCRIPT,
        "--labels-npy", label_path,
        "--out", vti_out
    ], check=True)

    # Call export_material.py
    subprocess.run([
        "python", MATERIAL_SCRIPT,
        "--props", prop_path,
        "--out", yaml_out
    ], check=True)

    print(f"[DONE] {vti_out} and {yaml_out} generated.")
