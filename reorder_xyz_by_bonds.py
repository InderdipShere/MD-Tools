#!/usr/bin/env python3

import argparse
import numpy as np
from collections import defaultdict, deque
import re

def parse_itp(itp_path):
    atom_order = []
    atom_types = []
    bonds = []
    bond_lengths = {}
    atom_section = False
    bond_section = False
    atom_names = {}
    atom_types_map = {}

    with open(itp_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                atom_section = 'atoms' in line.lower()
                bond_section = 'bonds' in line.lower()
                continue
            if atom_section and line and not line.startswith(';'):
                parts = line.split()
                idx = int(parts[0])
                atom_type = parts[1]
                name = parts[4]
                atom_order.append(name)
                atom_names[idx] = name
                atom_types_map[idx] = atom_type
                atom_types.append(atom_type)
            if bond_section and line and not line.startswith(';'):
                parts = line.split()
                a1, a2 = int(parts[0]), int(parts[1])
                # Try to get bond length if present (usually 4th column)
                try:
                    length = float(parts[3])
                except (IndexError, ValueError):
                    length = None
                bonds.append((a1, a2))
                bond_lengths[(a1, a2)] = length
                bond_lengths[(a2, a1)] = length

    return atom_order, atom_types, bonds, bond_lengths, atom_types_map

def read_xyz(xyz_path):
    with open(xyz_path) as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())
    atoms = []
    coords = []

    for line in lines[2:2+num_atoms]:
        parts = line.strip().split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    return atoms, np.array(coords)

def distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist

def build_bond_graph(coords, atoms, threshold=1.8):
    dist = distance_matrix(coords)
    n = len(atoms)
    graph = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] < threshold:
                graph[i].append(j)
                graph[j].append(i)
    return graph

def detect_molecules(graph, num_atoms_expected, total_atoms):
    visited = set()
    molecules = []

    for atom in range(total_atoms):
        if atom in visited:
            continue
        queue = deque([atom])
        mol = []
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            mol.append(curr)
            for neighbor in graph.get(curr, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(mol) >= num_atoms_expected - 1:
            molecules.append(mol)

    # Assign unconnected atoms (e.g., Cl⁻) to nearest molecule
    all_assigned = set(i for mol in molecules for i in mol)
    unassigned = set(range(total_atoms)) - all_assigned
    return molecules, unassigned

def assign_unbonded(unassigned, max_atom_mol, molecules):
    print("unassigned", unassigned)
    print("molecules", molecules)
    for atom in unassigned:
        for i, mol in enumerate(molecules):
            if(len(mol)<max_atom_mol):
                molecules[i].append(atom)

def reorder_molecules(molecules, atom_order, atom_types, template_adj, bond_lengths, atoms, coords, mol_graph, tol=0.3):
    """
    For each detected molecule (list of global atom indices) find a mapping from the
    template (atom_order, template_adj) to the actual atoms using atom names and
    bond connectivity/lengths. Treat template nodes with no template neighbors (unbonded,
    e.g. Cl) specially: map bonded scaffold first, then assign unbonded atoms to nearest
    available atoms in that molecule.
    Returns list of lists of global indices ordered by template.
    """
    reordered = []
    T = len(atom_order)

    for mol_idx, mol in enumerate(molecules):
        mol_set = set(mol)
        # restricted graph for this molecule
        mol_graph_restricted = {g: [n for n in mol_graph[g] if n in mol_set] for g in mol}
        # candidates for each template node (global indices)
        candidates = {t: [g for g in mol if atoms[g] == atom_order[t]] for t in range(T)}

        # identify unbonded template nodes (no neighbors in template)
        unbonded_t = [t for t in range(T) if len(template_adj[t]) == 0]
        bonded_t = [t for t in range(T) if t not in unbonded_t]

        # ordering of bonded template nodes for search: smallest candidate list first
        order = sorted(bonded_t, key=lambda t: (len(candidates[t]) if candidates[t] else 999, -len(template_adj[t])))

        assigned = [None] * T
        used = set()
        success = False

        def dfs(pos):
            nonlocal success
            if pos >= len(order):
                success = True
                return True
            t = order[pos]
            if not candidates[t]:
                return False
            for g in candidates[t]:
                if g in used:
                    continue
                # neighbor consistency checks
                ok = True
                for nb_t in template_adj[t]:
                    nb_assigned = assigned[nb_t]
                    if nb_assigned is None:
                        continue
                    # must be neighbors in molecule graph
                    if nb_assigned not in mol_graph_restricted.get(g, []):
                        ok = False
                        break
                    # check bond length if available
                    exp = bond_lengths.get((t+1, nb_t+1), None)
                    if exp is not None:
                        obs = np.linalg.norm(coords[g] - coords[nb_assigned])
                        if abs(obs - exp*10) > tol:
                            ok = False
                            break
                if not ok:
                    continue
                # pass, assign
                assigned[t] = g
                used.add(g)
                if dfs(pos+1):
                    return True
                assigned[t] = None
                used.remove(g)
            return False

        # attempt search on bonded scaffold
        if bonded_t and all(candidates[t] for t in bonded_t):
            dfs(0)

        if not success:
            print(f"Warning: No perfect mapping found for molecule {mol_idx+1}; falling back to greedy label matching for scaffold.")
            # greedy: fill bonded positions with unused atoms matching label, else any unused
            for t in bonded_t:
                if assigned[t] is None:
                    for g in candidates[t]:
                        if g not in used:
                            assigned[t] = g
                            used.add(g)
                            break
            for t in bonded_t:
                if assigned[t] is None:
                    for g in mol:
                        if g not in used:
                            assigned[t] = g
                            used.add(g)
                            break

        # assign unbonded template positions (e.g., Cl) by nearest available atom in molecule
        centroid = np.mean([coords[g] for g in mol], axis=0)
        for t in unbonded_t:
            # candidates within molecule with correct label and not used
            cands = [g for g in mol if atoms[g] == atom_order[t] and g not in used]
            if not cands:
                cands = [g for g in mol if g not in used]
            if not cands:
                assigned[t] = None
                continue
            # choose nearest to centroid (prefer spatially associated placement)
            chosen = min(cands, key=lambda g: np.linalg.norm(coords[g] - centroid))
            assigned[t] = chosen
            used.add(chosen)

        # Ensure all template positions are filled by using any remaining unused atoms
        remaining_atoms = [g for g in mol if g not in used]
        for t in range(T):
            if assigned[t] is None:
                if remaining_atoms:
                    g = remaining_atoms.pop(0)
                    assigned[t] = g
                    used.add(g)

        # finalize mapping in template order (preserve template length)
        mapped = [assigned[t] for t in range(T)]
        reordered.append(mapped)
    return reordered

def write_xyz(filename, atoms, coords, reordered_indices, atom_types=None, atom_order=None):
    def get_atom_label(idx, mol_idx=None, atom_types=None, atom_order=None):
        # If the atom index exceeds the template, use the actual atom name
        if atom_types is not None and mol_idx is not None and mol_idx < len(atom_types):
            return atom_types[mol_idx]
        elif atom_order is not None and mol_idx is not None and mol_idx < len(atom_order):
            return atom_order[mol_idx]
        else:
            return atoms[idx]
    with open(filename, 'w') as f:
        total = sum(len(mol) for mol in reordered_indices)
        f.write(f"{total}\n")
        f.write("Reordered structure\n")
        for mol_num, mol in enumerate(reordered_indices):
            for atom_num, i in enumerate(mol):
                if isinstance(i, (list, tuple, np.ndarray)):
                    # handle accidental nested structures by taking first element
                    i = int(i[0])
                # ensure i is scalar int
                i = int(i)
                label = get_atom_label(i, atom_num, atom_types, atom_order)
                f.write(f"{label}  {coords[i][0]:.6f}  {coords[i][1]:.6f}  {coords[i][2]:.6f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-itp', required=True, help='Input .itp file')
    parser.add_argument('-xyz', required=True, help='Input crystal .xyz file')
    parser.add_argument('-o', '--output', required=True, help='Output .xyz file')
    parser.add_argument('-at', choices=['yes', 'no'], default='no', help='Print atom type instead of atom name (default: no)')
    parser.add_argument('-debug', choices=['yes', 'no'], default='no', help='Print atom names for each molecule before reordering (default: no)')

    args = parser.parse_args()

    print("Parsing ITP file...")
    atom_order, atom_types, bonds, bond_lengths, atom_types_map = parse_itp(args.itp)
    atom_per_mol = len(atom_order)
    print(f"Expected atoms per molecule: {atom_per_mol}")

    print("Reading XYZ file...")
    atoms, coords = read_xyz(args.xyz)
    num_atoms = len(atoms)
    num_molecules = num_atoms // atom_per_mol

    print(f"Detected {num_molecules} molecules.")
    print("Reordering atoms in molecules using bond map and bond lengths...")

    # Build bond graph from coordinates
    graph = build_bond_graph(coords, atoms, threshold=1.9)  # You can adjust threshold for your system
    print("Detecting molecules by connectivity...")
    molecules, unassigned = detect_molecules(graph, atom_per_mol, num_atoms)
    print(f"Detected {len(molecules)} molecules by connectivity.")
    if args.debug == 'yes':
        print('--- DEBUG: Atom names for each molecule (by connectivity) ---')
        for m, mol in enumerate(molecules):
            mol_atoms = [atoms[i] for i in mol]
            print(f"Molecule {m+1}: {mol_atoms}")
        print('--- DEBUG: Atom order from ITP ---')
        print(f"ITP atom order: {atom_order}")
        print('-----------------------------------')

    # Assign unbonded atoms (Cl) to nearest molecule
    if unassigned:
        print(f"Assigning {len(unassigned)} unbonded atoms (like Cl) to molecules...")
        # Get list of unbonded Cl indices
        cl_indices = [i for i in unassigned if atoms[i] == 'Cl']
        # how many atoms each molecule needs to reach the template size
        deficits = [max(0, atom_per_mol - len(mol)) for mol in molecules]
        # assign at most deficits[m] Cls to molecule m; choose nearest available Cl
        remaining_cl = list(cl_indices)
        molecule_centroids = [np.mean(coords[mol], axis=0) for mol in molecules]
        for m_idx, deficit in enumerate(deficits):
            for _ in range(deficit):
                if not remaining_cl:
                    break
                # pick nearest remaining Cl to this molecule centroid
                cl_choice = min(remaining_cl, key=lambda c: np.linalg.norm(coords[c] - molecule_centroids[m_idx]))
                molecules[m_idx].append(cl_choice)
                if args.debug == 'yes':
                    print(f"DEBUG: Assigned Cl atom index {cl_choice} to molecule {m_idx+1}")
                remaining_cl.remove(cl_choice)
        # if any Cl left (more Cl than deficits), assign them by nearest-molecule as fallback
        for cl_idx in remaining_cl:
            nearest_mol = min(range(len(molecules)), key=lambda m: np.linalg.norm(coords[cl_idx] - molecule_centroids[m]))
            molecules[nearest_mol].append(cl_idx)
            if args.debug == 'yes':
                print(f"DEBUG: Assigned extra Cl atom index {cl_idx} to molecule {nearest_mol+1}")

    # Build template adjacency from ITP bonds
    T = len(atom_order)
    template_adj = [[] for _ in range(T)]
    for a1, a2 in bonds:
        i1, i2 = a1-1, a2-1
        if 0 <= i1 < T and 0 <= i2 < T:
            template_adj[i1].append(i2)
            template_adj[i2].append(i1)

    # Reorder atoms in each molecule according to ITP template using graph matching
    reordered_indices = reorder_molecules(molecules, atom_order, atom_types, template_adj, bond_lengths, atoms, coords, graph)

    print(f"Writing reordered XYZ file to: {args.output}")
    write_xyz(args.output, atoms, coords, reordered_indices, atom_types=atom_types if getattr(args, 'at', 'no') == 'yes' else None, atom_order=atom_order)
    print("Done.")

if __name__ == '__main__':
    main()
