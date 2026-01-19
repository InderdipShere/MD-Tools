#!/usr/bin/env python3
"""
Generate scattering intensities I(q) from provided structure factors S_AB(q) and
tabulated atomic form-factor parameters.

Inputs
------
-sf : structure factor file (.xvg/.dat) containing q and S_AB(q) columns
-atomic_file : atomic form-factor parameter file
-column_index : 1-based column indices to read from the structure factor file
-column_sf_alpha : atom names for each selected S_AB(q) column (alpha index)
-column_sf_beta  : atom names for each selected S_AB(q) column (beta index)
-column_alpha_Natoms : atom counts N_A for each selected column
-column_beta_Natoms  : atom counts N_B for each selected column

Outputs
-------
-aff : atomic form factors f_atom(q) for each unique atom type
-o   : I(q) and individual contributions f_A(q) f_B(q) sqrt(N_A N_B) S_AB(q)

Definitions
-----------
S_AB(q) = (1 / sqrt(N_A N_B)) < sin(q r_ij) / (q r_ij) >
I(q)    = sum_{A,B} f_A(q) f_B(q) sqrt(N_A N_B) S_AB(q)

Form factor per atom:
    f_atom(q) = sum_{k=1..4} a_k * exp(-b_k * (q/(4*pi))**2) + c
Parameters are read from the atomic_file in sections like:
    [ SOL ]
    OW  a1 b1 a2 b2 a3 b3 a4 b4 c
    HW1 a1 b1 a2 b2 a3 b3 a4 b4 c
"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sf", required=True, help="Structure factor file (.xvg/.dat) with q and S_AB(q)")
    #parser.add_argument("--q-column", type=int, default=1, help="1-based column index for q values (default: 1)")
    parser.add_argument("--column-index", required=True, nargs="+", type=int, help="1-based indices of S_AB(q) columns to use")
    parser.add_argument("--sf-alpha", required=True, nargs="+", help="Atom names (alpha) matching column_index order")
    parser.add_argument("--sf-beta", required=True, nargs="+", help="Atom names (beta) matching column_index order")
    parser.add_argument("--atomic-file", required=True, help="Atomic form-factor parameter file")
    parser.add_argument("--atomic-type", "-atm_type", nargs="+", required=True, help="Names of atom types (must include atoms referenced by --sf-alpha/--sf-beta)")
    parser.add_argument("--atomic-count", "-atm_count", nargs="+", type=int, required=True, help="Counts of atom types (same length as --atomic-type)")
    parser.add_argument("-o", "--output-file", default="intensity.xvg", help="Output file for q, I(q), and contributions")
    parser.add_argument("--ff-output", action="store_true", help="Optional output file for atomic form factors)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def parse_atomic_form_factors(path: str) -> Dict[str, List[float]]:
    """Return nested dict[group][atom] -> [a1,b1,a2,b2,a3,b3,a4,b4,c]."""
    data: Dict[str, Dict[str, List[float]]] = {}
    current = None
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].split(";", 1)[0].strip()
            if not line:
                continue            
            tokens = line.split()
            atom = tokens[0]
            params = list(map(float, tokens[1:]))
            if len(params) != 9:
                raise ValueError(f"Expected 9 numeric parameters for atom {atom} in group {current}, got {len(params)}")
            data[atom] = params
    return data


def calc_atomic_form_factor(q: np.ndarray, params: List[float]) -> np.ndarray:
    a1, b1, a2, b2, a3, b3, a4, b4, c = params
    # q in nm^-1 
    q=q*1e-1  # convert to Å^-1
    x = (q / (4.0 * np.pi)) ** 2
    return (
        a1 * np.exp(-b1 * x)
        + a2 * np.exp(-b2 * x)
        + a3 * np.exp(-b3 * x)
        + a4 * np.exp(-b4 * x)
        + c
    )


def load_structure_factor(sf_path: str, s_cols: List[int])-> Tuple[np.ndarray, np.ndarray]:
    """Load q and selected S_AB(q) columns (1-based indices)."""
    
    numeric_lines: List[str] = []
    with open(sf_path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith(('#', '@')):
                continue
            numeric_lines.append(line)

    if not numeric_lines:
        raise ValueError(f"No numeric data found in structure factor file: {sf_path}")

    data = np.loadtxt(io.StringIO("".join(numeric_lines)))
    if data.ndim == 1:
        data = data[None, :]
    s_indices = [c  for c in s_cols]
    q_values = data[:, 0]
    s_data = np.vstack([data[:, idx] for idx in s_indices])  # shape (n_selected, n_q)
    return q_values, s_data


def select_ff_section(ff_params: Dict[str, Dict[str, List[float]]], section: Optional[str]) -> Dict[str, List[float]]:
    if section:
        if section not in ff_params:
            available = ", ".join(ff_params.keys())
            raise KeyError(f"Section '{section}' not found in atomic form-factor file. Available: {available}")
        return ff_params[section]
    if len(ff_params) == 1:
        return next(iter(ff_params.values()))
    available = ", ".join(ff_params.keys())
    raise KeyError(f"Multiple sections found ({available}); specify one with --ff-section")


def main() -> None:
    args = get_cli_args()

    log_file = os.path.splitext(args.output_file)[0] + ".log"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )

    logging.info("Starting intensity calculation with:")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    
    
    # Load structure factor data    
    q_values, s_data = load_structure_factor(args.sf, args.column_index)
    s_data = np.array(s_data)
    logging.info(f"Loaded {q_values.size} q-points for {s_data.shape[0]} sf from {args.sf}")
    logging.info(f"Shape of s_data: {s_data.shape}")
    
    # Atomic form factors
    ff_section: Dict[str, List[float]] = {}
    ff_params_raw = parse_atomic_form_factors(args.atomic_file)    
    for atm, cont in zip(args.atomic_type, args.atomic_count):
        a_idx=args.atomic_type.index(atm)
        params_atm=ff_params_raw[atm]
        ff_section[a_idx]=calc_atomic_form_factor(q_values, params_atm)
        logging.info(f"{a_idx} Atom types: {atm}, =counts: {cont}")

    # Compute intensity
    I_total = np.zeros_like(q_values, dtype=np.float64)
    contributions: List[np.ndarray] = []
    labels: List[str] = []

    for idx in range(s_data.shape[0]):
        sab = s_data[idx, :]
        alpha = args.sf_alpha[idx]
        beta = args.sf_beta[idx]
        a_idx =args.atomic_type.index(alpha)
        b_idx =args.atomic_type.index(beta)
        Na = args.atomic_count[a_idx]
        Nb = args.atomic_count[b_idx]
        fa = ff_section[a_idx]
        fb = ff_section[b_idx]
        contrib = fa * fb * np.sqrt(Na * Nb) * sab
        I_total += contrib
        contributions.append(contrib)
        labels.append(f"S_{alpha}-{beta}")
        logging.debug(f"Added contribution for {alpha}-{beta} (Na={Na}, Nb={Nb})")
    # Save atomic form factors
    if args.ff_output:
        ff_output = os.path.splitext(args.output_file)[0] + "_ff.xvg"
        ff_header = "# q(nm^-1) " + " ".join(args.atomic_type)
        ff_data = np.column_stack([q_values] + [ff_section[a_idx] for a_idx in range(len(args.atomic_type))])
        np.savetxt(ff_output, ff_data, header=ff_header, fmt="%12.6f")
        logging.info(f"Wrote atomic form factors to {ff_output}")

    # Save intensity and contributions
    header = "# q(nm^-1) I(q) " + " ".join(labels)
    out_cols = [q_values, I_total] + contributions
    np.savetxt(args.output_file, np.column_stack(out_cols), header=header, fmt="%12.6f")
    logging.info(f"Wrote I(q) and contributions to {args.output_file}")


if __name__ == "__main__":
    main()
