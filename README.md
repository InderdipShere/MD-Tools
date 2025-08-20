# MD-Tools
Small codes to facilitate simulations

Code: reorder_xyz_by_bonds.py
Purpose: reorder atoms in a crystal .xyz that contains many copies of a single molecule so that each molecule’s atom order exactly matches the template in a GROMACS .itp (atom names and bond list).
Inputs: a GROMACS .itp file and an .xyz file with many molecules.
Output: a reordered .xyz where each molecule is laid out in the same atom order as the .itp.

Why it’s important?: 
GROMACS topology files expect atom ordering to match the topology (.itp/.top). If your coordinate file (e.g., .gro or .xyz) has different atom order, simulations will be wrong.
Automatically reordering many molecules prevents manual, error-prone fixes and allows you to apply the same .itp to periodic systems or crystals of repeated molecules.
How to use?:
Example: python reorder_xyz_by_bonds.py -xyz disordered.xyz -itp forcefield.itp -o ordered.xyz -at no -debug yes
CLI:
-itp required: path to .itp
-xyz required: input .xyz
-o/--output required: output .xyz
-at yes|no (default no): write atom type (OH) instead of atom name (O)
-debug yes|no (default no): print molecule lists and atom assignment details

Cautions & limitations:
itp should have only one molecule.
xyz file can have many molecule but of one type.
