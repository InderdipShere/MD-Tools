#!/usr/bin/env python3
"""
Generate Structure factors S(q) from provided radial distribution functions g_AB(r) 

Inputs
------
-rf : radial distribution function file (.xvg/.dat) containing r and g_AB(r) columns
-atomic_file : atomic form-factor parameter file
-column_index : 1-based column indices to read from the radial distribution function file
-rf_alpha : atom names for each selected g_AB(r) column (alpha index)
-rf_beta  : atom names for each selected g_AB(r) column (beta index)
-volumn : system volume in nm^3
-atomic-type : names of atom types 
-column_alpha_Natoms : atom counts N_A for each selected column
-column_beta_Natoms  : atom counts N_B for each selected column
-Rcut : maximum r value for integration (default: max r in rdf file)

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
from math import sin, sqrt
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.integrate import simpson
import numpy as np


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--rdf", required=True, help="RDF file (.xvg/.dat) with q and S_AB(q)")
    parser.add_argument("--column-index", required=True, nargs="+", type=int, help="1-based indices of S_AB(q) columns to use")
    parser.add_argument("--atm-alpha", required=True, nargs="+", help="Atom names (alpha) matching column_index order")
    parser.add_argument("--atm-beta", required=True, nargs="+", help="Atom names (beta) matching column_index order")
    parser.add_argument("--atm-cont-alpha", required=True, nargs="+", type=int, help="Atom counts (alpha) matching column_index order")
    parser.add_argument("--atm-cont-beta", required=True, nargs="+", type=int, help="Atom counts (beta) matching column_index order")
    parser.add_argument("--volume", "-vol", nargs="+", type=float, required=True, help="Counts of atom types (same length as --atomic-type)")
    parser.add_argument("--r_cut", type=float, default=None, help="Maximum r value for integration (default: max r in rdf file)")
    parser.add_argument("--q_max", type=float, default=200.0, help="Maximum q value for calculations (default: 200.0 nm^-1)")
    parser.add_argument("--q_bins", type=int, default=1000, help="Number of q bins for calculations (default: 1000)")
    parser.add_argument("-o", "--output-file", default="S_ab.xvg", help="Output file for S(q)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()

def load_rdf(rf_path: str, r_cols: List[int])-> Tuple[np.ndarray, np.ndarray]:
    """Load r and selected g_AB(r) columns (1-based indices)."""
    
    numeric_lines: List[str] = []
    with open(rf_path) as f:
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
    r_indices = [c  for c in r_cols]
    r_values = data[:, 0]
    rdf_data = np.vstack([data[:, idx] for idx in r_indices])  # shape (n_selected, n_r)
    return r_values, rdf_data

def calc_sf_4m_rdf(r: np.ndarray, rdf: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    """Calculate structure factor s(q) using RDF data."""
    hr= rdf - 1.0
    sf_integral = np.zeros_like(q_values, dtype=np.float64)
    for q_idx, q in enumerate(q_values):
        integrand = hr * r * np.sin(q * r) / q
        integral = simpson(y=integrand, x=r)
        sf_integral[q_idx] = integral
    return sf_integral    

def calc_sf_4m_rdf_ext(r: np.ndarray, rdf: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    """Calculate structure factor s(q) using RDF data."""
    hr= rdf - 1.0
    rcut= r[-1]
    pi_rcut= np.pi / rcut
    sf_integral = np.zeros_like(q_values, dtype=np.float64)
    for q_idx, q in enumerate(q_values):
        #integrand = hr * r * np.sin(q * r) / q
        integrand = hr * np.sin(q * r) * np.sin(pi_rcut *r )/ q        
        integral = simpson(y=integrand, x=r)
        sf_integral[q_idx] = integral
    return sf_integral    

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
    if len(args.volume) == 3:
        volume = args.volume[0] * args.volume[1] * args.volume[2]
    else:
        volume = args.volume[0] 
    logging.info(f"Using system volume: {volume} nm^3")
    r_values, rdf_data = load_rdf(args.rdf, args.column_index)
    if args.r_cut is not None:
        mask = r_values <= args.r_cut
        r_values = r_values[mask]
        rdf_data = rdf_data[:, mask]
        args.r_cut = r_values[-1]
        logging.info(f"Applied r_cut: {args.r_cut} nm {r_values[-1]}nm, resulting in {r_values.size} r-points")
    rdf_data = np.array(rdf_data)
    q_step = args.q_max/args.q_bins
    logging.info(f"Calculated q_step: {q_step} nm^-1")
    q_values = np.arange(q_step, args.q_max, q_step)
    logging.info(f"Loaded {r_values.size} r-points for {rdf_data.shape[0]} rdf from {args.rdf}")
    logging.info(f"Shape of rdf_data: {rdf_data.shape}")
    
   
    labels: List[str] = []
    sf_data = []

    for idx in range(rdf_data.shape[0]):
        alpha = args.atm_alpha[idx]
        beta  = args.atm_beta[idx]
        Na = args.atm_cont_alpha[idx]
        Nb = args.atm_cont_beta[idx]
        if alpha == beta and Na == Nb: #same inteaction
            dab = 1.0
        else:
            dab = 0.0        
        if(args.r_cut is None):
            sf_ab = calc_sf_4m_rdf(r_values, rdf_data[idx,:], q_values) 
            sf_ab *=4.0 * np.pi * sqrt(Na * Nb) / (volume)
        else:
            sf_ab = calc_sf_4m_rdf_ext(r_values, rdf_data[idx,:], q_values) 
            sf_ab *=4.0 *args.r_cut * sqrt(Na * Nb) / (volume)
        sf_ab += dab 
        sf_data.append(sf_ab)
        labels.append(f"S_{alpha}-{beta}")
        logging.debug(f"Added contribution for {alpha}-{beta} (Na={Na}, Nb={Nb})")
    # Save atomic form factors
    
    # Save intensity and contributions
    header = "# q(nm^-1)" + " ".join(labels)
    out_cols = [q_values] + sf_data
    np.savetxt(args.output_file, np.column_stack(out_cols), header=header, fmt="%12.6f")
    logging.info(f"Wrote S(q) to {args.output_file}")


if __name__ == "__main__":
    main()
