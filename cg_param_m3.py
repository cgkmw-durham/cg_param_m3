#!/usr/bin/env python

import os
import numpy as np
import itertools
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdchem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDConfig
import sys
import re
import math
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import collections
import random

def read_DG_data(DGfile):
    # Reads Delta G_OW for fragments into dictionary
    DG_data = {}
    with open(DGfile) as f:
        for line in f:
            (smi,DG,src) = line.rstrip().split()
            DG_data[smi] = {'DG':float(DG),'src':src}

    return DG_data

def include_weights(A,w):
    # Weights atoms by setting diagonal components
    A_weighted = np.copy(A)
    for i,weight in enumerate(w):
        A_weighted[i,i] = weight

    return A_weighted

def get_weights(groups,w_init,path_matrix):
    # Set weight of beads as average atomic mass * longest path
    w = []
    for node in groups:
        avgmass = get_avgmass(node,w_init)
        wi = avgmass * (get_size(node,path_matrix))
        w.append(wi)

    return w

def rank_nodes(A):
    # Ranks nodes by absolute eigenvector components of largest eigenvalue
    vals,vecs = np.linalg.eig(A)
    maxval = np.argmax(vals)
    scores = np.absolute(vecs[:,maxval])
    scores = scores/np.amin(scores)
    ranked = np.argsort(scores)

    ties = []
    sublist = []

    # Create ranked list with tied nodes at the same rank
    score_prev = scores[ranked[0]]
    for i in ranked:
        score_i = scores[i]
        if np.isclose(score_i,score_prev):
            sublist.append(i)
        else:
            ties.append(sublist)
            sublist = [i]
        score_prev = score_i
    ties.append(sublist)

    return scores,ties

def lone_atom(ties,A,A_init,scores,ring_beads,matched_maps,comp,exclusion_list):
    groups = []
    #Finds single-atom beads and takes atoms from adjacent beads
    temp_exclusions = []

    n = 0
    for rank in ties:
        for node in rank:
            if len(comp[node]) == 1:
                test_group = [comp[node][0]]
                temp_exclusions.append(test_group[0])

                # Bonded in final CG iteration
                connects = A[node]
                bonded = [i for i in np.nonzero(connects)[0] if not any(i in m for m in matched_maps)]
                bonded_scores = np.asarray([scores[bonded[k]] for k in range(len(bonded))])
                bonded_sorted = np.argsort(bonded_scores)
                for j,nbor in enumerate(bonded_sorted[:]):
                    if any(x in exclusion_list for x in comp[bonded[nbor]]):
                        bonded_sorted = np.delete(bonded_sorted,j)
                        bonded_scores = np.delete(bonded_scores,j)

                # Bonded in AA rep
                aa_connects = A_init[comp[node][0]]
                aa_bonded = [i for i in np.nonzero(aa_connects)[0]]

                #Steal atoms from most central neighbours most 'central' neighbours
                stolen_from = []
                score_prev = scores[bonded[bonded_sorted[0]]]
                for j in bonded_sorted:
                    scorej = scores[bonded[j]]
                    if np.isclose(scorej,score_prev):
                        stolen_from.append(bonded[j])
                        # For 2-atom beads at ends of molecules, just add whole bead
                        if len(comp[bonded[j]]) == 2 and len(np.nonzero(A[bonded[j]])[0]) == 1:
                            test_group.extend(comp[bonded[j]])
                        elif any(np.size(np.intersect1d(comp[bonded[j]],ring)) != 0 for ring in ring_beads):
                            test_group.extend(comp[bonded[j]])
                        else:
                            for a in aa_bonded:
                                if a in comp[bonded[j]]:
                                    test_group.append(a)
                groups.append(test_group)
                n = len(groups) - 1
                #Remove atoms from original groups
                for k in stolen_from:
                    test_group = comp[k][:]
                    for b in comp[k]:
                        if b in groups[n]:
                            test_group.remove(b)
                    if test_group != []:
                        groups.append(test_group)

    exclusion_list.extend(temp_exclusions)
    temp_groups = groups[:]
    
    for a in range(3):
        new_nodes = []
        for group in temp_groups:# check for shared atoms in new groups, and combine if there are any
            for i in range(len(new_nodes)):
                if np.size(np.intersect1d(group,new_nodes[i])) != 0:
                    new_nodes[i] = [int(p) for p in np.unique(np.concatenate((group,new_nodes[i]),axis=None)).tolist()]
                    break
            else:
                new_nodes.append([int(p) for p in group])
        temp_groups = new_nodes[:]

    #Put remaining groups back in
    for bead in comp:
        if not any(any(atom in group for group in groups) for atom in bead):
            new_nodes.append(bead)

    #Make new ring_beads
    new_ring_beads = []
    for ring in ring_beads:
        #ring_comp in comp[ring]
        for i,group in enumerate(new_nodes): 
            if any(atom in ring for atom in group) and i not in new_ring_beads:
                new_ring_beads.append(i)

    return new_nodes,ring_beads,exclusion_list


def spectral_grouping(ties,A,scores,ring_beads,comp,path_matrix,max_size,matched_maps):
    #Carries out an iteration of the spectral graph-based mapping scheme
    groups = []

    # Loop through ranks and apply spectral mapping scheme
    for rank in ties:
        new_groups = []
        for node in rank:
            # Prevents ring beads from combining with each other
            if any(node in a for a in groups) or any(node in a for a in ring_beads) or any(node in m for m in matched_maps):
                continue 
            # Get list of nodes connected to current node (with equal or lower rank)
            test_group = [node]
            connects = A[node]
            bonded = [i for i in np.nonzero(connects)[0] if not (any(i in a for a in groups) or any(i in a for a in matched_maps))]
            bonded_scores = np.asarray([scores[bonded[k]] for k in range(len(bonded))])
            bonded_sorted = np.argsort(bonded_scores)
            
            # Combine with most similar bonded node
            for j in bonded_sorted:
                #scorej = scores[bonded[j]]
                if test_group == [node]:
                    if any(bonded[j] in a for a in ring_beads):
                        continue
                    if get_size(comp[node] + comp[bonded[j]],path_matrix) <= max_size: # Prevent beads from getting to large
                        scorej = scores[bonded[j]]
                        test_group.append(bonded[j])
                    else:
                        break
                elif np.isclose(scorej,scores[bonded[j]]): # Combine all with same score (redundant)
                    test_group.append(bonded[j])
                else:
                    break
            new_groups.append(test_group)
        new_nodes = []
        for group in new_groups:# check for shared atoms in new groups, and combine if there are any
            for i in range(len(new_nodes)):
                if np.size(np.intersect1d(group,new_nodes[i])) != 0:
                    new_nodes[i] = np.unique(np.concatenate((group,new_nodes[i]),axis=None)).tolist()
                    break
            else:
                new_nodes.append(group)

        #Reverse combination of nodes if the size limit is exceeded
        for k in new_nodes[:]:
            compk = []
            for atom in k:
                compk.extend(comp[atom])

            if get_size(compk,path_matrix) > max_size:
                new_nodes.remove(k)
                for x in k:
                    new_nodes.append([x])
        groups = groups + new_nodes
    groups,ring_beads,matched_maps = process_rings(ring_beads,matched_maps,groups)# Tidy up ring-specific things

    return groups,ring_beads,matched_maps

def process_rings(ring_beads,matched_maps,groups):

    # If ring-bead not already in a bead, add as its own bead
    for bead in ring_beads:
        if not any(any(a in group for a in bead) for group in groups):
            groups.append(bead)
    
    for match in matched_maps:
        groups.append(match)
    # If bead includes part of a ring bead, add rest of ring bead
    for i in range(len(groups)):
        for bead in ring_beads:
            if np.size(np.intersect1d(bead,groups[i])) != 0:
                groups[i] = np.unique(np.concatenate((groups[i],bead),axis=None)).tolist()

    # Combine beads which share a ring bead (happens for multi-substituent beads)
    new_groups = []
    for l in range(len(groups)):
        if any(any(atom in bead for bead in new_groups) for atom in groups[l]):
            continue
        new_group = groups[l][:]
        for m in range(len(groups)):
            if np.size(np.intersect1d(new_group,groups[m])) != 0:
                new_group = np.unique(np.concatenate((new_group,groups[m]),axis=None)).tolist()
        new_groups.append(new_group)
    
    groups = new_groups 

    new_ring_beads = []
    new_matched_maps = []

    # Copy ring-containing beads to new ring-bead list
    for k,group in enumerate(groups):
        for j in range(len(ring_beads)):
            if any(a in ring_beads[j] for a in group):
                new_ring_beads.append([k])
                break
        for p in range(len(matched_maps)):
            if any(a in matched_maps[p] for a in group):
                new_matched_maps.append([k])
                break

    return groups,new_ring_beads,new_matched_maps

def new_connectivity(groups,oldA):
    # Get A matrix for new mapping
    newA = np.zeros((len(groups),len(groups)),dtype=int)
    for i,gi in enumerate(groups):
        for j,gj in enumerate(groups[i+1:]):
            for k in gi:
                for l in gj:
                    if oldA[k,l] == 1:
                        newA[i,i+j+1] = 1
                        newA[i+j+1,i] = 1
                if newA[i,i+j+1] == 1:
                    break

    return newA

def iteration(results,itr,A_init,w_init,ring_beads,path_matrix,matched_maps):
    results_dict = dict.fromkeys(['A','comp'])

    # Get properties of current mapping
    if itr == 0:
        oldA = np.copy(A_init)
        comp = [[i] for i in range(len(w_init))]
        w = w_init[:]
    else:
        oldA = results[itr-1]['A']
        comp = results[itr-1]['comp']
        w = get_weights(comp,w_init,path_matrix)
        
    A_weighted = include_weights(oldA,w)

    # Get new mapping scheme
    scores,ties = rank_nodes(A_weighted)
    groups,ring_beads,matched_maps = spectral_grouping(ties,oldA,scores,ring_beads,comp,path_matrix,3,matched_maps)
    results_dict['A'] = new_connectivity(groups,oldA)

    # Get atomistic composition of new mapping
    if itr == 0:
        results_dict['comp'] = groups[:]

    else:
        comp = []
        for gj in groups:
            bead_comp = list(itertools.chain.from_iterable([results[itr-1]['comp'][x] for x in gj]))
            comp.append(bead_comp)

        results_dict['comp'] = comp[:]


    return results_dict,ring_beads,matched_maps

def group_rings(A,ring_atoms,matched_maps,moli):
    # Pre-processing step for ring structures    

    new_groups = []

    #List of possible edge fragments ordered by size, with mappings
    edge_frags = collections.OrderedDict()
    edge_frags["[R1][R1][R1][R1][R1][R1]"] =  [[0,1],[2,3],[4,5]]
    edge_frags["[R1][R1][R1][R1][R1]"] = [[0,1,2],[2,3]]
    edge_frags["[R1][R1][R1][R1]"] = [[0,1],[2,3]]
    edge_frags["[R1][R1][R1]"] =  [[0,1,2]]
    edge_frags["[R1][R1]"] = [[0,1]]
    edge_frags["[R2][R1][R2]"] = [[0,1,2]]

        #Map edges first
    for substruct in edge_frags:
        #matches = fragment.GetSubstructMatches(Chem.MolFromSmarts(substruct))
        matches = moli.GetSubstructMatches(Chem.MolFromSmarts(substruct)) 

        for match in matches:
            all_shared = False
            for system in ring_atoms:
                if all(m in system for m in match):
                    all_shared = True
                    break
            if not all_shared:
                continue
            if substruct == "[R2][R1][R2]":
                overlap = False
                for matchj in matches:
                    if match != matchj:
                        if list(set(match).intersection(matchj)):
                            overlap = True
                            break
                if overlap:
                    continue
            for bead in edge_frags[substruct]:
                test_bead = [match[x] for x in bead]
                if not any(any(y in ngroup for ngroup in new_groups) for y in test_bead):
                    new_groups.append(test_bead)

    #Get remaining unmapped atoms 
    unmapped = []
    for ring in ring_atoms:
        for a in ring:
            if not any(a in group for group in new_groups):
                unmapped.append(a)
    #Mapping of unmapped fragments
    if unmapped:
        #Split into continous fragments
        unm_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli,unmapped)
        unm_smi = unm_smi.upper()
        unm_mol = Chem.MolFromSmiles(unm_smi)
        unmapped_frags = Chem.GetMolFrags(unm_mol)
        for frag in unmapped_frags:
            #Do mapping for each continuous fragment
            indices = [unmapped[k] for k in frag]
            frag_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli,unmapped)
            frag_smi = frag_smi.upper()
            frag_mol = Chem.MolFromSmiles(frag_smi) 
            A_frag = np.asarray(Chem.GetAdjacencyMatrix(frag_mol))

            #Check if there are complete rings within unmapped fragments
            frag_ring_atoms = get_ring_atoms(frag_mol)
            if frag_ring_atoms:
                new_beads = group_rings(A_frag,frag_ring_atoms,matched_maps,frag_mol)[1]
            else:
                new_beads = []
            frag_ring_beads = new_beads[:]

            #Apply on iteration of graph-based mapping
            if sum([len(b) for b in frag_ring_beads]) < len(frag): 
                path_frag = floyd_warshall(csgraph=A_frag,directed=False)
                w_frag = [1.0 for atom in frag_mol.GetAtoms()] 
                A_fragw = include_weights(A_frag,w_frag)
                scores,ties = rank_nodes(A_fragw)
                comp = [[i] for i in range(frag_mol.GetNumAtoms())]            
         
                new_beads.extend(spectral_grouping(ties,A_frag,scores,frag_ring_beads,comp,path_frag,2,matched_maps)[0])
            for bead in new_beads:
                new_groups.append([indices[x] for x in bead])
              
    ring_beads = new_groups[:]
    # Add non-ring atoms
    new_groups += matched_maps
    for i in range(A.shape[0]):
        if not any(i in a for a in new_groups):
            new_groups.append([i])

    return ring_beads,new_groups,A                 
 
def postprocessing(results,ring_atoms,n_iter,A_init,w_init,path_matrix,matched_maps):
    #Checks if overall mapping is too high resolution
    last_iter = results[n_iter -1]
    exclusion_list = []
    postprocess = 1
    while postprocess:
        min_size = 1000 
        avg_size = 0
        count = 0.0
        for i,bead in enumerate(last_iter['comp']):
            size = len(bead)
            if size < min_size:
                min_size = size
            if i not in ring_atoms:
                avg_size += size
                count += 1.0
        avg_size = avg_size / count
    
        if min_size == 1:
            postprocess = 1
        else:
            postprocess = 0

        if postprocess:
            #Applies a path contraction if there are single-atom beads
            results_dict,ring_atoms,exclusion_list= path_contraction(last_iter,postprocess,A_init,w_init,ring_atoms,matched_maps,path_matrix,exclusion_list)
        else:
            results_dict = last_iter.copy()
        last_iter = results_dict.copy()
 
    return results_dict


def path_contraction(last_iter,postprocess,A_init,w_init,ring_beads,matched_maps,path_matrix,exclusion_list):
    #Applies a path contraction
    results_dict = dict.fromkeys(['A','comp'])

    oldA = last_iter['A']
    comp = last_iter['comp']
    w = get_weights(comp,w_init,path_matrix)

    A_weighted = include_weights(oldA,w)

    scores,ties = rank_nodes(A_weighted)
    groups,ring_beads,exclusion_list = lone_atom(ties,oldA,A_init,scores,ring_beads,matched_maps,comp,exclusion_list)
    results_dict['A'] = new_connectivity(groups,A_init)

    results_dict['comp'] = groups[:]
    return results_dict,ring_beads,exclusion_list


def get_size(comp,path_matrix):
    
    # Find longest path between atoms in bead
    longpath = 0
    for i in comp:
        for j in comp:
            path = path_matrix[i,j]
            if path > longpath:
                    longpath = path

    return longpath

def get_avgmass(comp,masses):
    #Average atomic mass of heavy atoms in bead
    avgmass = sum([masses[i] for i in comp])/len(comp)
    return avgmass
    
def get_paths(A_atom,mol):
    
    #Gets shortest path between each pair of atoms
    dist_matrix,preds = floyd_warshall(csgraph=A_atom,directed=False,return_predecessors=True)
    n_atoms = len(mol.GetAtoms())

    #Doubly weights atoms in 3rd row of periodic table
    row_weights = []
    for at in mol.GetAtoms():
        if at.GetAtomicNum <= 10:
            row_weights.append(1)
        else:
            row_weights.append(2)

    path_matrix = np.zeros((dist_matrix.shape()))

    #Gets path lengths in terms of weighted atom sums
    for i in range(n_atoms-1):
        for j in range(i,n_atoms):
            min_path = 0
            node = j
            while node != i:
                min_path += row_weights[node]
                node = preds[i,node]
            min_path += row_weights[i]
            path_matrix[i,j] = min_path

    return path_matrix
                


def mapping(mol,ring_atoms,matched_maps,n_iter):
    #Initialise data structures
    #mol = Chem.MolFromSmiles(smiles)
    A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol))
    path_matrix = floyd_warshall(csgraph=A_atom,directed=False)
    w_init = [atom.GetMass() for atom in mol.GetAtoms()]
    #w_init = [1.0 for atom in mol.GetAtoms()]
    ring_beads,comp,A_init = group_rings(A_atom,ring_atoms,matched_maps,mol)
    #w_init = get_weights(comp,w_init)

    # Do spectral mapping iterations
    results = []
    for itr in range(n_iter):
        results_dict,ring_beads,matched_maps = iteration(results,itr,A_init,w_init,ring_beads,path_matrix,matched_maps)
        results.append(results_dict)
 

    # Get final mapping
    results_dict_final = postprocessing(results,ring_atoms,n_iter,A_init,w_init,path_matrix,matched_maps)
    #sizes = get_sizes(results[best]['comp'],A_init)
    ring_beads = []
    for ring in ring_atoms:
        cgring = []
        for atom in ring:
            for i,bead in enumerate(results_dict_final['comp']):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        ring_beads.append(cgring)

    return results_dict_final['A'],results_dict_final['comp'],ring_beads,path_matrix#,sizes

def get_ring_atoms(mol):
    #get ring atoms and systems of joined rings 

    rings = mol.GetRingInfo().AtomRings()
    ring_systems = []
    for ring in rings:
        ring_atoms = set(ring)
        new_systems = []
        for system in ring_systems:
            shared = len(ring_atoms.intersection(system))
            if shared:
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        ring_systems = new_systems

    return [list(ring) for ring in ring_systems]
        

def get_hbonding(mol,beads):
    #Extracts h-bonding behaviour for all beads in molecule
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
 
    h_donor = []
    h_acceptor = []
    for feat in feats:
        if feat.GetFamily() == "Donor":
            for i in feat.GetAtomIds():
                for b,bead in enumerate(beads):
                    if i in bead:
                       if b not in h_donor:
                           h_donor.append(b)
                       break
        if feat.GetFamily() == "Acceptor":
            for i in feat.GetAtomIds():
                for b,bead in enumerate(beads):
                    if i in bead:
                       if b not in h_acceptor:
                           h_acceptor.append(b)
                       break

    return h_donor,h_acceptor

def get_smi(bead,mol):
    #gets fragment smiles from list of atoms

    bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead)

    #Work out aromaticity by looking for lowercase c and heteroatoms
    ring_size = 0
    frag_size = 0
    lc = re.compile('[cn([nH\])os]+')
    lc = string_lst = ['c','\\[nH\\]','(?<!\\[)n','o']
    lowerlist = re.findall(r"(?=("+'|'.join(string_lst)+r"))",bead_smi)
    
    #Construct test rings for aromatic fragments
    if lowerlist:
        frag_size = len(lowerlist)
        #For two atoms + substituents, make a 3-membered ring
        if frag_size == 2:
            subs = bead_smi.split(''.join(lowerlist))
            for i in range(len(subs)):
                if subs[i] != '':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}cc1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
            ring_size = 6
            if not Chem.MolFromSmiles(bead_smi): #If fragment isn't kekulisable use 5-membered ring
                bead_smi = 'c1c{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
                ring_size = 5
        #For three atoms + substituents, make a dimer
        elif len(lowerlist) == 3:
            split1 = bead_smi.split(''.join(lowerlist[:2]))
            split2 = split1[1].split(lowerlist[2])
            subs = [split1[0],split2[0],split2[1]]
            for i in range(len(subs)):
                if subs[i] != '' and subs[i][0] != '(':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])       
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)

            ring_size = 6
            if not Chem.MolFromSmiles(bead_smi):
                bead_smi = 'c1{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])
                ring_size = 5

    if not Chem.MolFromSmiles(bead_smi):
        bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
        ring_size = 0
        frag_size = 0

    #Standardise SMILES for lookup
    bead_smi = Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(bead_smi))
        
    return bead_smi,ring_size,frag_size
    
def get_types(beads,mol,ring_beads):
    #loops through beads and determines bead type
    script_path = os.path.dirname(os.path.realpath(__file__))
    DG_data = read_DG_data('{}/fragments-exp.dat'.format(script_path))
    bead_types = []
    charges = []
    all_smi = []
    h_donor,h_acceptor = get_hbonding(mol,beads)
    for i,bead in enumerate(beads):
        qbead = sum([mol.GetAtomWithIdx(int(j)).GetFormalCharge() for j in bead])
        charges.append(qbead)
        bead_smi,ring_size, frag_size = get_smi(bead,mol)
        all_smi.append(bead_smi)
        bead_types.append(param_bead(bead,bead_smi,ring_size,frag_size,any(i in ring for ring in ring_beads),qbead,i in h_donor,i in h_acceptor,DG_data))

    return(bead_types,charges,all_smi,DG_data)

def get_diffs(alogps,ring_size,frag_size,category,size):
    #Gets free energy differences between fragment and all bead types
    #if ring_size == 0:
    #    delta_Gs = np.array([-9.2,-9.1,-7.4,-5.1,-3.8,-2.0,-1.1,0.0,2.2,1.8,5.6,8.1,10.1,11.2,13.4,13.8,14.8,18.9])
    ## Redo these for Martini 3
    ## Note, these sizes only include ring atoms, not substituents
    #elif ring_size - frag_size == 4:
    #    delta_Gs = np.array([-0.76,-1.99,-1.17,3.57,4.96,9.27,10.58,13.06,16.42,18.05,19.34,19.71])
    #elif ring_size - frag_size == 3:
    #    delta_Gs = np.array([-3.21,-4.27,-2.98,1.81,3.55,8.06,10.14,12.89,16.43,18.39,19.87,20.52])
    #elif ring_size - frag_size == 2:
    #    delta_Gs = np.array([-5.26,-6.63,-5.47,-0.64,1.02,5.69,7.49,9.96,13.54,15.52,16.86,17.43])
    #else:
    #    print("No free energy data for that fragment-ring combination")
    #    exit()
    # Set of DGs from delta_Gs[ring-frag][category][size]
    delta_Gs = {
        0:{
            'standard':{
                'T': [-14.8,-15.2,-12.1,-9.8,-8.8,-7.2,-6.1,-4.9,-2.9,-3.1,0.3,2.3,3.6,4.5,6.4,6.7,7.8,12.0],
                'S': [-12.0,-11.8,-9.8,-7.7,-6.9,-5.2,-4.2,-3.6,-0.9,-1.8,2.1,3.6,5.3,6.3,8.4,9.2,9.9,14.2],
                'R': [-9.2,-9.1,-7.4,-5.1,-3.8,-2.0,-1.1,0.0,2.2,1.8,5.6,8.1,10.1,11.2,13.4,13.8,14.8,18.9]
            }
           # 'da':{
           #     'T': [-12.7,-13.2,-9.5,-7.8,-6.8,-5.0,-4.1,-2.8,-1.2,-1.4,2.3,3.9],
           #     'S': [-9.6,-9.5,-7.8,-6.1,-5.4,-3.7,-2.5,-1.0,1.1,0.2,3.8,6.0],
           #     'R': [-7.4,-7.0,-5.1,-3.5,-1.9,0.2,1.0,2.2,4.3,3.8,7.8,10.7]   
           # }
        },
        4:{
            'standard':{
                'T': [-5.23,-5.77,-3.77,-0.35,0.44,2.18,2.90,4.08,6.03,5.41,8.92,10.64,11.84,12.56,14.35,14.74,15.74,19.10],
                'S': [-3.89,-4.15,-1.84,-0.08,0.78,2.39,3.31,4.20,6.66,5.84,9.78,11.56,12.84,13.99,16.20,16.56,17.32,20.91],
                'R': [-4.27,-4.01,-1.64,0.26,1.66,3.55,4.53,5.43,7.93,7.43,11.49,13.79,15.63,16.77,18.90,19.61,20.59,24.01]
            }
        #    'da':{
        #        'T': [-12.7,-13.2,-9.5,-7.8,-6.8,-5.0,-4.1,-2.8,-1.2,-1.4,2.3,3.9],
        #        'S': [-9.6,-9.5,-7.8,-6.1,-5.4,-3.7,-2.5,-1.0,1.1,0.2,3.8,6.0],
        #        'R': [-7.4,-7.0,-5.1,-3.5,-1.9,0.2,1.0,2.2,4.3,3.8,7.8,10.7]
        #    }
        },
        3:{},2:{}}

    diffs = np.abs(np.array(delta_Gs[ring_size-frag_size][category][size]) - alogps)

    return diffs

def param_bead(bead,bead_smi,ring_size,frag_size,ring,qbead,don,acc,DG_data):
    #Parametrises bead type
    types = ['P6','P5','P4','P3','P2','P1','N6','N5','N4','N3','N2','N1','C6','C5','C4','C3','C2','C1']

    #Get h-bonding category
    #if acc or don:
    #    category = 'da'
    #    if acc and don:
    #        suffix = 'da'
    #    elif acc:
    #        suffix = 'a'
    #    elif don:
    #        suffix = 'd'
    #else:
    category = 'standard'
    suffix = ''

    path_length = get_size(bead,path_matrix)#path_length counts bonds spanning fragment
   # if ring_size != 0:
   #     if frag_size == 2:#frag_size counts only atoms on the ring
   #         size = 'T'
   #         prefix = 'T'
   #     elif frag_size == 3:
   #         size = 'S'
   #        prefix = 'S'

    #Get bead sizes from path length regardless of ring status
    if path_length == 1:
        size = 'T'
        prefix = 'T'
    elif path_length == 2:
        size = 'S'
        prefix = 'S'
    else:
        size = 'R'
        prefix = ''

    #Check for SMARTS matches
    btype = ''
    for m,match in enumerate(matched_maps):
        if sorted(match) == sorted(bead):
            btype = matched_beads[m]

    if btype == '':
        #Parametrise charged beads based on h-bonding behaviour
        if qbead != 0:
            if acc and don:
                btype = 'Qda'
            elif acc:
                btype = 'Qa'
            elif don:
                btype = 'Qd'
            else:
                btype = 'Q0'

        else:
            try:
                #Get from list of precalculated fragments
                alogps = DG_data[bead_smi]['DG']
            except:
                #If not on list, get from server or Wildmann-Crippen
                print('{} not on list'.format(bead_smi))
                alogps = get_alogps(bead_smi)

            

            #Get difference between fragment DG_OW and all beads
            diffs = get_diffs(alogps,ring_size,frag_size,category,size)

            #If close to Nda bead, parametrise from h-bonding behaviour
            #if diffs[5] == 1.0 and (acc or don):
            #    if acc and don:
            #        btype = 'Nda'
            #    elif acc:
            #        btype = 'Na'
            #    elif don:
            #        btype = 'Nd'
            #Otherwise, pick bead with closest DG_OW
            #else:
            sort_diffs = np.argsort(diffs)
            btype = types[sort_diffs[0]]

    btype = prefix + btype + suffix
    #Ring beads are S type
    #if ring:
    #    btype = 'S' + btype

    return btype                        


def get_alogps(bead_smi):
    #Gets ALOGPS value from server. If this fails for whatver reason, use Wildmann-Crippen
    try:
        alogps = requests.get('http://vcclab.org/web/alogps/calc?SMILES=' + bead_smi).text
    except:
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        print(bead_smi,'Data from Wildmann-Crippen')
        return logK*5.74
    if 'error' not in alogps:
        logK = float(alogps.split()[4])
    else:
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        print(bead_smi,'Data from Wildmann-Crippen')
    
    return logK*5.74

def bead_coords(bead,conf):
    #Get coordinates of a bead

    coords = np.array([0.0,0.0,0.0])
    total = 0.0
    for atom in bead:
        mass = mol.GetAtomWithIdx(atom).GetMass() 
        coords += conf.GetAtomPosition(atom)*mass
        total += mass
    coords /= (total*10.0)

    return coords

def write_gro(mol_name,bead_types,coords0,gro_name):
    #write gro file
    conf = mol.GetConformer(0)
    with open(gro_name,'w') as gro:
        gro.write('single molecule of {}\n'.format(mol_name))
        gro.write('{}\n'.format(len(bead_types)))
        i = 1
        for bead,xyz in zip(bead_types,coords0):
            gro.write('{:5d}{:5}{:>5}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(1,mol_name,bead,i,xyz[0],xyz[1],xyz[2]))
            i += 1
        gro.write('5.0 5.0 5.0')

def get_virtual_sites(ring,coords,A_cg):
    #Get projection of ring beads onto a plane, and define real sites as outer
    # hull, virtual sites as inner beads
   
    #Set up coordinate list for ring system
    coords_r = np.empty((len(ring),3))
    for i,a in enumerate(ring):
        coords_r[i] = coords[a]

    #Center on the origin
    com = np.sum(coords_r,axis=0)/coords_r.shape[0]
    coords_c = np.subtract(coords_r,com)

    #Build inertia tensor
    I_xx = sum([(c[1]**2 + c[2]**2) for c in coords_c])
    I_yy = sum([(c[0]**2 + c[2]**2) for c in coords_c])
    I_zz = sum([(c[0]**2 + c[1]**2) for c in coords_c])
    I_xy = -sum([(c[0]*c[1]) for c in coords_c])
    I_xz = -sum([(c[0]*c[2]) for c in coords_c])
    I_yz = -sum([(c[1]*c[2]) for c in coords_c])
    I = np.array([[I_xx,I_xy,I_xz],[I_xy,I_yy,I_yz],[I_xz,I_yz,I_zz]])

    # Get vectors on plane (two smallest principal axes)
    Ivals,Ivecs = np.linalg.eig(I)
    Isort = np.argsort(Ivals)
    plane_x = Ivecs[:,Isort[0]]
    plane_y = Ivecs[:,Isort[1]]
    

    #Project points onto new coordinates
    coords_p = np.empty((coords_c.shape[0],2))
    for i,coord in enumerate(coords_c):
        coords_p[i][0] = np.dot(plane_x,coord)
        coords_p[i][1] = np.dot(plane_y,coord)

    #No virtual sites if only 3 sites in ring system
    if len(ring) <= 3:
        real_sites = [r for r in ring]
        virtual_sites = []
    #Get convex hull and set real sites
    else:
        hull = ConvexHull(coords_p)
        verts = hull.vertices
        real_sites = [ring[j] for j in verts]
        virtual_sites = [site for site in ring if site not in real_sites]

    #Check if any inner beads are bonded to beads outside the ring system, and make these real sites
    for vs in list(virtual_sites):
        bonded = [j for j in np.nonzero(A_cg[vs])[0]]
        rvs = coords[vs]
        for b in bonded:
            if b not in ring:
                virtual_sites.remove(vs)
                min_v = 100000
                closest = 0
                #Find closest edge in convex hull
                for e in range(len(real_sites)):
                    #Project vs onto edge
                    ra = coords[real_sites[e]]
                    rb = coords[real_sites[(e+1)%(len(real_sites))]]
                    rab = np.subtract(rb,ra)
                    rav = np.subtract(rvs,ra)
                    rproj = np.add(ra,(np.dot(rab,rav)/np.dot(rab,rab))*rab)
                    dist = np.linalg.norm(np.subtract(rvs,rproj))
                    if dist < min_v:
                        closest = e
                        min_v = dist
                #Insert between vertices defining closest edge
                real_sites.insert((closest+1)%len(real_sites),vs)
                break

    vs_weights = {}
    for vs in virtual_sites:
        vs_weights[vs] = (construct_vs(ring.index(vs),verts,coords_p,ring))#Inputs in ring frame of reference

    return real_sites,vs_weights

def construct_vs(vs,real_sites,coords_p,ring):
    #Constructs virtual sites as linear combination of 4 nearest real sites (or 3 if there are only 3)
    dists = [np.linalg.norm(coords_p[vs]-coords_p[rs]) for rs in real_sites]
    weights = {}
    vx,vy = coords_p[vs]

    if len(real_sites) >= 4:
        closest = np.argsort(dists)[:4]
        vertices = [real_sites[r] for r in range(len(real_sites)) if r in closest]
        r1x,r1y = coords_p[vertices[0]]
        r2x,r2y = coords_p[vertices[3]]
        r3x,r3y = coords_p[vertices[1]]
        r4x,r4y = coords_p[vertices[2]]
        tx = r4x + r1x -r3x - r2x
        ty = r4y + r1y - r3y - r2y
        c = ((r1y-vy)*(r3x-r1x) - (r1x-vx)*(r3y-r1y))
        b = (r2y-r1y)*(r3x-r1x) + (r1y-vy)*tx - (r2x-r1x)*(r3y-r1y) - (r1x-vx)*ty
        a = (r2y-r1y)*tx - (r2x-r1x)*ty
        roots = np.roots([a,b,c])

        for f in roots:
            if (f >= 0.0 and f <= 1.0) or np.isclose(f,1.0) or np.isclose(f,0.0):
                f1 = f
                break
        f2 = -( (r1x-vx) + f1*(r2x-r1x)) / ( (r3x-r1x) + f1*tx)

        weights = {}
        weights[ring[vertices[0]]] = (1-f1)*(1-f2)
        weights[ring[vertices[3]]] = f1*(1-f2)
        weights[ring[vertices[1]]] = (1-f1)*f2
        weights[ring[vertices[2]]] = f1*f2

    elif len(real_sites) == 3:
        vertices = real_sites[:]
        r1x,r1y = coords_p[vertices[0]]
        r2x,r2y = coords_p[vertices[1]]
        r3x,r3y = coords_p[vertices[2]]

        M = np.array([[(r2x-r1x),(r3x-r1x)],[(r2y-r1y),(r3y-r1y)]])
        B = np.array([(vx-r1x),(vy-r1y)])
        P = np.linalg.solve(M,B)

        weights[ring[vertices[1]]] = P[0]
        weights[ring[vertices[2]]] = P[1]
        weights[ring[vertices[0]]] = 1.0 - P[0] - P[1]

    return weights


def ring_bonding(real,virtual,A_cg,dihedrals):
    #Constructs constraint structure for ring systems
    
    #Remove all bonds from virtual sites
    for vs in list(virtual.keys()):
        for i in range(A_cg.shape[0]):
            A_cg[vs,i] = 0
            A_cg[i,vs] = 0

    #Construct outer frame
    A_cg[real[0],real[-1]] = 1
    A_cg[real[-1],real[0]] = 1
    for r in range(len(real)-1):
        A_cg[real[r],real[r+1]] = 1
        A_cg[real[r+1],real[r]] = 1
    
    #Construct inner frame and hinge dihedrals
    n_struts = len(real)-3
    j = len(real)-1
    k = 1
    struts = 0
    for s in range(int(math.ceil(n_struts/2.0))):
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = (j+1)%len(real) #First one loops round to 0
        l = k+1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        k += 1
        if struts == n_struts:
            break
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = k-1
        l = j-1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        j -= 1

    return A_cg,dihedrals
        

def get_masses(bead_types,virtual_sites):
    #Get masses, including setting virtual sites to 0
    masses = []
    for b,bead in enumerate(bead_types):
        if bead[0] == 'S':
            if b in virtual_sites:
                masses.append(0.0)
            else:
                masses.append(45.0)
        else:
            masses.append(72.0)

    return masses
            

def write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,itp_name):
    #writes gromacs topology file
    with open(itp_name,'w') as itp:
        itp.write('[moleculetype]\n')
        itp.write('MOL    2\n')
        virtual,real = write_atoms(itp,A_cg,mol_name,bead_types,charges,all_smi,coords0,ring_beads)
        bonds,constraints,dihedrals = write_bonds(itp,A_cg,ring_beads,real,virtual)
        angles = write_angles(itp,bonds,constraints)
        if dihedrals:
            write_dihedrals(itp,dihedrals,coords0)
        if virtual:
            write_virtual_sites(itp,virtual)

def write_atoms(itp,A_cg,mol_name,bead_types,charges,all_smi,coords,ring_beads):
    #Writes [atoms] block in itp file
    real = []
    virtual = {}
    #Split ring beads into real and virtual sites
    for ring in ring_beads:
        rs,vs = get_virtual_sites(ring,coords,A_cg)
        virtual.update(vs)
        real.append(rs)

    masses = get_masses(bead_types,virtual)

    itp.write('\n[atoms]\n')
    
    for b in range(len(bead_types)):
    #    if DG_data[all_smi[b]]['src'] == 'E':
    #        DG_src = 'Experiment'
    #    elif DG_data[all_smi[b]]['src'] == 'A':
    #        DG_src = 'ALOGPS'
        itp.write('{:5d}{:>5}{:5d}{:>5}{:>5}{:5d}{:>10.3f}{:>10.3f};{}\n'.format(b+1,bead_types[b],1,mol_name,'CG'+str(b+1),b+1,charges[b],masses[b],all_smi[b]))

    return virtual,real
    
def write_bonds(itp,A_cg,ring_atoms,real,virtual):
    #Writes [bonds] and [constraints] blocks in itp file
    #Construct bonded structures for ring systems, including dihedrals   
    dihedrals = []
    for r,ring in enumerate(ring_atoms):
        A_cg,dihedrals = ring_bonding(real[r],virtual,A_cg,dihedrals)

    itp.write('\n[bonds]\n')
    bonds = [list(pair) for pair in np.argwhere(A_cg) if pair[1] > pair[0]]
    constraints = []
    k = 1250.0

    #Get average bond lengths from all conformers
    rs = np.zeros(len(bonds))
    coords = np.zeros((len(beads),3))
    for conf in mol.GetConformers():
        for i,bead in enumerate(beads):
            coords[i] = bead_coords(bead,conf)
        for b,bond in enumerate(bonds):
            rs[b] += np.linalg.norm(np.subtract(coords[bond[0]],coords[bond[1]]))/nconfs

    #Split into bonds and constraints, and write bonds
    con_rs = []
    for bond,r in zip(bonds,rs):
        share_ring = False
        for ring in ring_atoms:
            if bond[0] in ring and bond[1] in ring:
                share_ring = True
                constraints.append(bond)
                con_rs.append(r)
                break
        if not share_ring:
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(bond[0]+1,bond[1]+1,1,r,k))

    #Write constraints
    if len(constraints) > 0:
        itp.write('\n#ifdef min\n')
        k = 5000000.0
        for con,r in zip(constraints,con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(con[0]+1,con[1]+1,1,r,k))

        itp.write('\n#else\n')
        itp.write('[constraints]\n')
        for con,r in zip(constraints,con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}\n'.format(con[0]+1,con[1]+1,1,r))
        itp.write('#endif\n')

    return bonds,constraints,dihedrals

def write_angles(itp,bonds,constraints):
    #Writes [angles] block in itp file
    k = 25.0

    #Get list of angles
    angles = []
    for bi in range(len(bonds)-1):
        for bj in range(bi+1,len(bonds)):
            shared = np.intersect1d(bonds[bi],bonds[bj])
            if np.size(shared) == 1:
                if bonds[bi] not in constraints or bonds[bj] not in constraints:
                    x = [i for i in bonds[bi] if i != shared][0]
                    z = [i for i in bonds[bj] if i != shared][0]
                    angles.append([x,int(shared),z])
    #Calculate and write to file
    if angles:
        itp.write('\n[angles]\n')
        coords = np.zeros((len(beads),3))
        thetas = np.zeros(len(angles))
        print(thetas)
        for conf in mol.GetConformers():
            for i,bead in enumerate(beads):
                coords[i] = bead_coords(bead,conf)
            for a,angle in enumerate(angles):
                vec1 = np.subtract(coords[angle[0]],coords[angle[1]])
                vec1 = vec1/np.linalg.norm(vec1)
                vec2 = np.subtract(coords[angle[2]],coords[angle[1]])
                vec2 = vec2/np.linalg.norm(vec2)
                theta = np.arccos(np.dot(vec1,vec2))
                print(theta)
                thetas[a] += (theta*180.0)/(np.pi*nconfs)
                #print(vec1,vec2)

        #thetas = thetas*180.0/(np.pi)
        print(thetas)


        for a,t in zip(angles,thetas):
            itp.write('{:5d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(a[0]+1,a[1]+1,a[2]+1,2,t,k))


def write_dihedrals(itp,dihedrals,coords0):
    #Writes hinge dihedrals to itp file 
    #Dihedrals chosen in ring_bonding
    itp.write('\n[dihedrals]\n')
    k = 500.0

    for dih in dihedrals:
        vec1 = np.subtract(coords0[dih[1]],coords0[dih[0]])
        vec2 = np.subtract(coords0[dih[2]],coords0[dih[1]])
        vec3 = np.subtract(coords0[dih[3]],coords0[dih[2]])
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = vec2/np.linalg.norm(vec2)
        vec3 = vec3/np.linalg.norm(vec3)
        cross1 = np.cross(vec1,vec2)
        cross1 = cross1/np.linalg.norm(cross1)
        cross2 = np.cross(vec2,vec3)
        cross2 = cross2/np.linalg.norm(cross2)
        angle = np.arccos(np.dot(cross1,cross2))*180.0/np.pi
        itp.write('{:5d}{:3d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(dih[0]+1,dih[1]+1,dih[2]+1,dih[3]+1,2,angle,k))

def write_virtual_sites(itp,virtual_sites):
    #Write [virtual_sites] block to itp file
    itp.write('\n[virtual_sitesn]\n')
    
    vs_iter = sorted(virtual_sites.keys())

    for vs in vs_iter:
        cs = sorted(virtual_sites[vs].items())
        if len(cs) == 4:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs+1,3,cs[0][0]+1,cs[0][1],cs[1][0]+1,cs[1][1],cs[2][0]+1,cs[2][1],cs[3][0]+1,cs[3][1]))
        elif len(cs) == 3:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs+1,3,cs[0][0]+1,cs[0][1],cs[1][0]+1,cs[1][1],cs[2][0]+1,cs[2][1]))
    
    itp.write('\n[exclusions]\n')
    
    done = []

    #Add exclusions between vs and all other beads
    for vs in vs_iter:
        excl = str(vs+1)
        for i in range(len(beads)):
            if i != vs and i not in done:
                excl += ' '+str(i+1)
        done.append(vs)
        itp.write('{}\n'.format(excl))

smi = sys.argv[1]    
mol_name = 'MOL'

def get_coords(mol,beads):
    #Calculates coordinates for output gro file
    mol_Hs = Chem.AddHs(mol)
    conf = mol_Hs.GetConformer(0)

    cg_coords = []
    for bead in beads:
        coord = np.array([0.0,0.0,0.0])
        total = 0.0
        for atom in bead:
            mass = mol.GetAtomWithIdx(atom).GetMass()
            coord += conf.GetAtomPosition(atom)*mass
            total += mass
        coord /= (total*10.0)
        cg_coords.append(coord)

    cg_coords_a = np.array(cg_coords)

    return cg_coords_a

def get_smarts_matches(mol):
    #Get matches to SMARTS strings
    smarts_strings = {
    #'S([O-])(=O)(=O)O'  :    'Qa',
    #'S([O-])(=O)(=O)[C;!$(*F)]'   :    'Q0'
    #'C(=O)O' : 'P1'
    #'CC' : 'C2',
    #'OO' : 'P5'
    #'CCC' : 'C2',
    #'CCCC': 'C2'
    }
    ## Add function to get rid of groups with duplicate atoms 
    matched_maps = []
    matched_beads = []
    for smarts in smarts_strings:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        for match in matches:
            matched_maps.append(list(match))
            matched_beads.append(smarts_strings[smarts])

    return matched_maps,matched_beads

#Generate molecule object
smi = sys.argv[1]
mol_name = 'MOL'
mol = Chem.MolFromSmiles(smi)

#Coarse-grained mapping
matched_maps,matched_beads = get_smarts_matches(mol)
ring_atoms = get_ring_atoms(mol)
A_cg,beads,ring_beads,path_matrix = mapping(mol,ring_atoms,matched_maps,3)
non_ring = [b for b in range(len(beads)) if not any(b in ring for ring in ring_beads)]

#Parametrise beads
bead_types,charges,all_smi,DG_data = get_types(beads,mol,ring_beads)

#Generate atomistic conformers
nconfs = 1
mol = Chem.AddHs(mol)
AllChem.EmbedMultipleConfs(mol,numConfs=nconfs,randomSeed=random.randint(1,1000),useRandomCoords=True)
AllChem.UFFOptimizeMoleculeConfs(mol)
coords0 = get_coords(mol,beads)

#Calculate bonded interactions and write gromacs files
write_gro(mol_name,bead_types,coords0,sys.argv[2])
write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,sys.argv[3])
