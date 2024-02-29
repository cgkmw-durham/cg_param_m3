# cg_param_m3
Coarse-grained mapping and parametrisation for the Martini 3 forcefield. 

Run with: ./cg_param_m3 "[SMILES]" [name].gro [name].itp [dimer tuning (0/1)]

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

# Publication

This work has been published. For citation, please use the following reference: 

T.D. Potter, N. Haywood, A. Teixeira, G. Hodges, E.L. Barrett and M.A. Miller, Partitioning into phosphatidylcholine–cholesterol membranes: liposome measurements, coarse-grained simulations, and implications for bioaccumulation, Environ. Sci.: Processes Impacts, 2023, 25, 1082-1093 

A full description of the mapping and parametrisation procedures, originally developed for the Martini 2 force field, can be found in the following paper:

T.D. Potter, E.L. Barrett and M.A. Miller, Automated Coarse-Grained Mapping Algorithm for the Martini Force Field and Benchmarks for Membrane–Water Partitioning, J. Chem. Theory Comput., 2021, https://doi.org/10.1021/acs.jctc.1c00322.


