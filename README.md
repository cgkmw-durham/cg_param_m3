# cg_param_m3

Coarse-grained mapping and parametrisation for the Martini 3 forcefield
Origin: Miller's cg_param (J. Chem. Theory Comput. 2021, 17, 9, 5777–5791).

Modified the script to generate the itp file with optional Molecule name.

Run with: ./cg_param_m3 "[SMILES]" [name].gro [name].itp [dimer tuning (y/n)]

Outputs .gro and .itp files compatible with Gromacs

## Installation

Dependencies:

*numpy/scipy
*RDKit
*requests

The easiest way to install these dependencies is with conda

~~~~
$ conda create -c rdkit -n cg_param rdkit numpy scipy requests
$ conda activate cg_param
~~~~

## Run
```bash
conda activate cg_param
python cg_param_m3 "[SMILES]" [name].gro [name].itp name y    # Turn on parameters tuning
python cg_param_m3 "[SMILES]" [name].gro [name].itp name n    # Turn off parameters tuning
```

Please cite the original paper (https://pubs.acs.org/doi/10.1021/acs.jctc.1c00322)
and this project if you used cg_param_m3 in your work.
