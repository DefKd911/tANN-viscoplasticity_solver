#!/usr/bin/env python3
"""
export_material.py

Generate a DAMASK-compatible material YAML file from props_seedXXX.npy.

Usage:
    python export_material.py --props props_seed123.npy --out material_seed123.yaml
"""

import numpy as np
import yaml
import argparse
from typing import Dict


def elastic_constants_from_E_nu(E: float, nu: float) -> Dict[str, float]:
    """
    Convert isotropic E, nu to (C11, C12, C44) for a cubic Hooke representation.
    """
    C11 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C12 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C44 = E / (2.0 * (1.0 + nu))
    return {"C_11": float(C11), "C_12": float(C12), "C_44": float(C44)}


def export_material(props_file: str, out_file: str):
    # Load properties: expecting a dict-like object with keys: "E","nu","xi0","h0"
    props = np.load(props_file, allow_pickle=True).item()
    if not all(k in props for k in ("E", "nu", "xi0", "h0")):
        raise ValueError("props file must contain keys: 'E','nu','xi0','h0'")

    E_arr = np.asarray(props["E"], dtype=float)
    nu_arr = np.asarray(props["nu"], dtype=float)
    xi0_arr = np.asarray(props["xi0"], dtype=float)
    h0_arr = np.asarray(props["h0"], dtype=float)

    K = E_arr.size
    if not (nu_arr.size == K == xi0_arr.size == h0_arr.size):
        raise ValueError("All property arrays must have the same length K")

    # --- homogenization: one entry PER material index (N_constituents = 1) ---
    # Use simple 'pass' homogenization for single-constituent points.
    homogenization = {}
    for i in range(K):
        label = f"h{i}"
        homogenization[label] = {
            "N_constituents": 1,
            "mechanical": {
                "type": "pass",
                "output": ["F", "P", "stress_Cauchy"]
            },
            "thermal": {
                "type": "pass"
            }
        }

    # --- material: This must be a LIST. Each list entry corresponds to a material index used in the geometry.
    material_list = []
    for i in range(K):
        label = f"h{i}"
        entry = {
            "homogenization": label,
            "constituents": [
                {
                    "phase": f"grain_{i}",
                    "v": 1.0,
                    "O": [1.0, 0.0, 0.0, 0.0]  # identity quaternion (no rotation)
                }
            ]
        }
        material_list.append(entry)

    # --- phase definitions ---
    phase_dict = {}
    for i in range(K):
        E_i = float(E_arr[i])
        nu_i = float(nu_arr[i])
        xi0_i = float(xi0_arr[i])
        h0_i = float(h0_arr[i])

        C = elastic_constants_from_E_nu(E_i, nu_i)

        phase = {
            "lattice": "cF",  # assumed cubic; change if needed
            "mechanical": {
                "elastic": {
                    "type": "Hooke",
                    # DAMASK's Hooke expects stiffness components; provide basic isotropic cubic form
                    "C_11": C["C_11"],
                    "C_12": C["C_12"],
                    "C_44": C["C_44"],
                },
                # isotropic plasticity minimal config; adjust fields if your DAMASK build expects different names
                "plastic": {
                    "type": "isotropic",
                    "output": ["xi"],

                    # keys: xi_0, xi_inf, h_0 are commonly used by isotropic models in DAMASK examples
                    "xi_0": xi0_i,
                    "xi_inf": 1e12,
                    "h_0": h0_i,

                    # additional numeric defaults (safe values)
                    "dot_gamma_0": 1e-3,
                    "n": 20.0,
                    "a": 2.0,
                    "M": 1.0,
                    "h": 1.0,
                    "dilatation": True
                }
            }
        }

        phase_dict[f"grain_{i}"] = phase

    # --- Compose root YAML ---
    root = {
        "homogenization": homogenization,
        "material": material_list,
        "phase": phase_dict,
    }

    # Dump YAML to file (pretty)
    with open(out_file, "w") as f:
        yaml.safe_dump(root, f, sort_keys=False, default_flow_style=False)

    print(f"[export_material] wrote '{out_file}' with {K} material entries and {K} phases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DAMASK material YAML from props.npy")
    parser.add_argument("--props", required=True, help="path to props_seedXXX.npy (dict with E,nu,xi0,h0)")
    parser.add_argument("--out", default="", help="output yaml filename (optional)")
    parser.add_argument("--out-dir", default="material_yaml_fixed", help="directory to place generated YAMLs when --out not provided")
    args = parser.parse_args()
    out_path = args.out
    if not out_path:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.basename(args.props)
        stem = base.replace("props_", "").replace(".npy", "")
        if not stem.startswith("seed"):
            stem = f"seed{stem}"
        out_path = os.path.join(args.out_dir, f"material_{stem}.yaml")
    export_material(args.props, out_path)
