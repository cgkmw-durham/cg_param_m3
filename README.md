# cg_param_m3
Coarse-grained mapping and parametrisation for the Martini 3 forcefield. 

This version supersedes branch cg_param_m3/martini3_v2 in this repository.

Run with: ./cg_param_m3.py -s "[SMILES]" -f [NAME]

An explanation of the available options can be shown using: ./cg_param_m3.py -h

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

This software is associated with the following publication, in preparation:

A. Gredelj, J. Roberts, E. Kearney, E.L. Barrett, N. Haywood, D. Sheffield, G. Hodges and M.A. Miller, Predicting aquatic toxicity of anionic hydrocarbon and perfluorinated surfactants using membrane-water partition coefficients from coarse-grained simulations

For existing publications in the development of cg_param, please cite the following reference:

T.D. Potter, N. Haywood, A. Teixeira, G. Hodges, E.L. Barrett and M.A. Miller, Partitioning into phosphatidylcholine–cholesterol membranes: liposome measurements, coarse-grained simulations, and implications for bioaccumulation, Environ. Sci.: Processes Impacts, 2023, 25, 1082-1093 https://doi.org/10.1039/D3EM00081H

A full description of the mapping and parametrisation procedures in cg_param, originally developed for the Martini 2 force field, can be found in the following paper:

T.D. Potter, E.L. Barrett and M.A. Miller, Automated Coarse-Grained Mapping Algorithm for the Martini Force Field and Benchmarks for Membrane–Water Partitioning, J. Chem. Theory Comput., 2021, 17, 5777-5791 https://doi.org/10.1021/acs.jctc.1c00322.


