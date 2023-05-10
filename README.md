# cg_param_m3
-----------------Outdated version of code: See default branch for most recent release---------------

Coarse-grained mapping and parametrisation for the Martini 3 forcefield

Run with: ./cg_param_m3 "[SMILES]" [name].gro [name].itp [dimer tuning (0/1)]

Outputs .gro and .itp files compatible with Gromacs

This version implements the methodology described in our upcoming paper on diesters

Dependencies:

*numpy/scipy
*RDKit
*requests

The easiest way to install these dependencies is with conda

~~~~
$ conda create -c rdkit -n cg_param rdkit numpy scipy requests
$ conda activate cg_param
~~~~
