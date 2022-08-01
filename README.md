# cg_param_m3
Coarse-grained mapping and parametrisation for the Martini 3 forcefield

Run with: ./cg_param_m3 "[SMILES code]" [name].gro [name].itp [Tuning (0/1)]

Outputs .gro and .itp files compatible with Gromacs

Dependencies:

*numpy/scipy
*RDKit
*requests

The easiest way to install these dependencies is with conda

~~~~
$ conda create -c rdkit -n cg_param rdkit numpy scipy requests
$ conda activate cg_param
~~~~
