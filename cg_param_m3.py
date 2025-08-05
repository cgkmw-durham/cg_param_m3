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
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import sys
import re
import math
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import collections
import random
import matplotlib.pyplot as plt
from operator import itemgetter
import argparse
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
delta_Gs = {
    0:{
        'standard':{
            'T': [-14.8,-15.2,-12.1,-9.8,-8.8,-7.2,-6.1,-4.9,-2.9,-3.1,0.3,2.3,3.6,4.5,6.4,6.7,7.8,12.0],
            'S': [-12.0,-11.8,-9.8,-7.7,-6.9,-5.2,-4.2,-3.6,-0.9,-1.8,2.1,3.6,5.3,6.3,8.4,9.2,9.9,14.2],
            'R': [-9.2,-9.1,-7.4,-5.1,-3.8,-2.0,-1.1,0.0,2.2,1.8,5.6,8.1,10.1,11.2,13.4,13.8,14.8,18.9]
        },
        'halogen':{
            'T': [2.7,5.4,5.2,7.6],
            'S': [4.3,8.0,7.2,9.4],
            'R': [8.7,13.9,12.7,14.3]
        }#,
#        'da':{
#            'T': [-12.7,-13.2,-9.5,-7.8,-6.8,-5.0,-4.1,-2.8,-1.2,-1.4,2.3,3.9],
#            'S': [-9.6,-9.5,-7.8,-6.1,-5.4,-3.7,-2.5,-1.0,1.1,0.2,3.8,6.0],
#            'R': [-7.4,-7.0,-5.1,-3.5,-1.9,0.2,1.0,2.2,4.3,3.8,7.8,10.7]
#        }
    },
    4:{
        'standard':{
            'T': [-5.23,-5.77,-3.77,-0.35,0.44,2.18,2.90,4.08,6.03,5.41,8.92,10.64,11.84,12.56,14.35,14.74,15.74,19.10],
            'S': [-3.89,-4.15,-1.84,-0.08,0.78,2.39,3.31,4.20,6.66,5.84,9.78,11.56,12.84,13.99,16.20,16.56,17.32,20.91],
            'R': [-4.27,-4.01,-1.64,0.26,1.66,3.55,4.53,5.43,7.93,7.43,11.49,13.79,15.63,16.77,18.90,19.61,20.59,24.01]
        }#,
#        'da':{
#            'T': [-3.26,-3.78,-0.36,1.60,2.30,4.18,4.79,5.99,7.53,7.04,10.56,12.18],
#            'S': [-1.67,-1.71,-0.13,1.48,2.45,4.29,5.09,6.84,8.82,8.05,11.59,13.41],
#            'R': [-2.00,-1.61,0.18,2.03,3.54,5.31,6.42,8.00,9.80,13.73,16.17]
#        }
    },
    3:{
        'standard':{
            'T': [-7.73,-8.25,-6.19,-2.63,-1.85,0.12,0.80,2.12,4.12,3.66,7.13,8.93,10.11,10.85,12.71,12.92,13.99,17.66],
            'S': [-5.45,-5.57,-3.22,-1.59,-0.61,0.99,1.87,2.82,5.28,4.46,8.36,10.01,11.38,12.53,14.61,15.07,15.78,19.58],
            'R': [-4.68,-4.33,-2.10,-0.27,1.13,3.05,4.04,4.93,7.32,6.90,10.69,12.92,14.77,15.83,18.11,18.61,19.73,23.19]
        }
    },
    2:{
        'standard':{
            'T': [-10.22,-10.74,-8.61,-4.98,-4.09,-2.10,-1.33,0.09,1.97,1.62,5.16,6.99,8.26,9.04,10.98,11.23,16.19],
            'S': [-7.64,-7.69,-5.34,-3.62,-2.57,-0.92,-0.10,0.90,3.47,2.51,6.57,8.29,9.61,10.85,12.95,13.49,14.21,18.13],
            'R': [-6.43,-6.10,-3.79,-1.94,-0.50,1.42,2.40,3.30,5.83,5.27,9.22,11.67,13.39,14.43,16.75,17.37,18.41,22.03]
        }}}

m3_beads = {
    'standard': ['P6','P5','P4','P3','P2','P1','N6','N5','N4','N3','N2','N1','C6','C5','C4','C3','C2','C1'],
    'halogen': ['X4','X3','X2','X1']#,
    #'da': ['P6','P5','P4','P3','P2','P1','N6','N5','N4','N3','N2','N1']
    }

preset_beads = {
    'CC':'TC2',
    'CCC':'SC2',
#    'O=CO':'SP2',    #Parameterisations for esters from diester paper
    'CC(=O)O':'SN5',
    'CCC(=O)O':'N4',
    'COC=O':'N6',
    'COC(C)=O':'N4'
    #'CCO' : 'SP2', #Experimental ether parameterisations, apply with care
    #'COC' : 'SP2'
    #'CCOC' : 'P2',
    #'COCC' : 'P2',
    #'CCOC' : 'P4',
    }

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

def Scale_Scores(A):

    # Dynamically scale scores to avoid massive variation in eigenvectors (driven by sulfates i.e. CCCCCCCCCCOCCOCCCOS(=O)(=O)[O-])
    # This results in improper comparison, resulting in mappings varying between SMILES codes
    for n in range(1,10):
        vals,vecs = np.linalg.eig(A)
        maxval = np.argmax(vals)
        scores = vecs[:,maxval]
        # Ensure scores are properly converged (all same sign) and vary over a reasonable number of orders of magnitude
        if all(scores<0) or all(scores>0):
            scores=np.absolute(scores)
            if np.absolute(math.log10(scores[np.argmin(scores)]))-np.absolute(math.log10(scores[np.argmax(scores)])) <10:
                #If scores are acceptable, end loop and return scores
                break

        # If loop not broken, divide adjacency function weights by 2 and try again
        np.fill_diagonal(A, A.diagonal() / 2)
    
    return(scores)

def rank_nodes(A):
 
    scores = Scale_Scores(A)        
    if args.v:print("",    
                    "    Bead Centrality Scores: ",
                    "",    
                    "    ",scores,    
                    "")    
    
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

    if args.v:print("")    
    if args.v:print("    Bead Centrality Scores: ")
    if args.v:print("")    
    if args.v:print("    ",scores)    
    if args.v:print("")    
    return scores,ties

def lone_atom(ties,A,A_init,scores,ring_beads,matched_maps,comp,exclusion_list):
    #Finds single-atom beads and takes atoms from adjacent beads
    groups = []
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

    #Verbose
    if args.v:
        print("    Present Atom Groupings: ","\033[38;5;34m", comp,"\033[0;0m", "Numbered as Bead(s): ", "\033[38;5;128m", list(range(len(comp))), "\033[0;0m")
        print("    Beads Grouped by Centrality Scores (Used for Assigning Order of Pairing): ","\033[38;5;128m",ties, "\033[0;0m")
    
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
    
    if args.v: print("    Adjacency Matrix: Weights of each bead on the diagonals, bonds indicated by off diagonal matrix values")
    if args.v: print('\t' + str(A_weighted).replace('\n', '\n\t'))
   
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
    if args.v:print("Ring Atom Groupings: ", "\033[38;5;34m",new_groups, "\033[0;0m")
    #Get remaining unmapped atoms 
    unmapped = []
    for ring in ring_atoms:
        for a in ring:
            if not any(a in group for group in new_groups):
                unmapped.append(a)


    #Mapping of unmapped fragments
    if unmapped:
        if args.v:print(" ")
        if args.v:print("Mapping Non-ring Atoms Via Speactral Mapping Algorithm: Note atoms renumbered from one for each branch which is mapped")
        if args.v:print(" ")

        #Split into continous fragments
        unm_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli,unmapped,canonical=False)
        unm_smi = unm_smi.upper()
        unm_mol = Chem.MolFromSmiles(unm_smi)
        unmapped_frags = Chem.GetMolFrags(unm_mol)

#        #Reordered unmapped based on connectivity
#        unmapped_new=[]
#        #Index 0 is not in the output smiles. Add if relevent. Assumes numbering by connectivity will always start at 0
#        if 0 in unmapped:
#            unmapped_new.append(0)
#        for atom in unm_smi.split(":")[1:]:
#            unmapped_new.append(int(atom.split("]")[0]))
#        unmapped=unmapped_new

        if args.v:print("Unmapped fragments: ", "\033[38;5;34m", unmapped_frags, "\033[0;0m")
        for frag in unmapped_frags:
            if args.v:print(    "Mapping Branch: ", "\033[38;5;34m", frag, "\033[0;0m")
            if args.v:print(" ")
            #Do mapping for each continuous fragment
            indices = [unmapped[k] for k in frag]
            frag_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli,unmapped).split(".")[0]
            frag_smi = frag_smi.upper()
            frag_mol = Chem.MolFromSmiles(frag_smi) 
            
            #Assign atom map so that subfrags can be reassigned. 
            assign_atom_maps(frag_mol)
            #Find atom map assignments (could also call using mol.GetAtomIDx) to allow backmapping for fragment. Add 0 if relevant, as this is not printed in the SMILES by defualt
            core_map=re.findall(r"\:([^\]]*)\]",frag_smi)
            if len(core_map) != len(indices):
                core_map.insert(0,0)

            A_frag = np.asarray(Chem.GetAdjacencyMatrix(frag_mol),dtype='f')
            #Check if there are complete rings within unmapped fragments
            frag_ring_atoms = get_ring_atoms(frag_mol)
            
            #print("frag_ring_atoms",frag_ring_atoms)
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
                if matched_maps:
                    #for i in matched_maps:
                    match=False
                    for n in matched_maps:
                        sorted_match=sorted(n)
                        sorted_bead=sorted(bead)
                        if sorted_match ==sorted_bead:
                            match=True
                    if not(match): 
                    #if any([sorted(i) in sorted(bead) for i in matched_maps]): 
                        #if sorted(bead) != sorted(i): 
                        #Indices based on number of atoms in fragment, not connectivity. Changed to allow represenation of large ring systems i.e. c1ccc6c(c1)Cc7c2ccccc2c8Cc3ccccc3c9c5cc4ccccc4cc5c6c7c89 
                        #new_groups.append([indices[x] for x in bead])
                        new_groups.append([int(core_map[x]) for x in bead])
                else:
                    #new_groups.append([indices[x] for x in bead])
                    new_groups.append([int(core_map[x]) for x in bead])
#        if args.v:print("    New Atom Groupings: ","\033[38;5;34m",results_dict["comp"],"\033[0;0m")
        if args.v:print(" ")
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
                
def assign_atom_maps(mol_dict):

    #Assign atom maps, allowing indexes to be passed from major fragments to subfragments.
    for atom in mol_dict.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
        # atom.SetProp('original_index', str(atom.GetIdx()))
    return mol_dict

def mapping(mol,ring_atoms,matched_maps,n_iter,mol_dict):
    #Initialise data structures
    #mol = Chem.MolFromSmiles(smiles)
    A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol),dtype='f')
    path_matrix = floyd_warshall(csgraph=A_atom,directed=False)
    w_init = [0.5*atom.GetMass() for atom in mol.GetAtoms()] #0.5 to avoid magnitude of eigenmatrixes getting too large during spectral mapping but large enough to differenetiate O, C and N
    #w_init = [1.0 for atom in mol.GetAtoms()]
    assign_atom_maps(mol_dict)
    ring_beads,comp,A_init = group_rings(A_atom,ring_atoms,matched_maps,mol_dict)

    # Do spectral mapping iterations
    results = []
    for itr in range(n_iter):
        
        if args.v:print(" ")
        if args.v:print("Mapping Iteration:", itr)
        if args.v:print(" ")

        #if args.v and results_dict:print(    "Mapping so far: ", "\033[38;5;34m", results_dict, "\033[0;0m")
        #if args.v:print(" ")
        
        results_dict,ring_beads,matched_maps = iteration(results,itr,A_init,w_init,ring_beads,path_matrix,matched_maps)
        results.append(results_dict)
        if args.v:print("    New Atom Groupings: ","\033[38;5;34m",results_dict["comp"],"\033[0;0m")
        if args.v:print(" ")

    if args.v:print('    Mapping Before Lone Atom Handling:',"\033[38;5;34m",results_dict['comp'],"\033[0;0m")

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
        #print("feat: ",feat.GetFamily(), feat.GetType(),feat.GetAtomIds())
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

    bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,canonical=True)

    if args.v: print("Bead Atoms and Smiles: ","\033[38;5;34m",bead,"\033[0;0m",bead_smi)

    #Work out aromaticity by looking for lowercase c and heteroatoms
    ring_size = 0
    frag_size = 0
    lc = re.compile('[cn([nH\\])os]+')
    lc = string_lst = ['c','\\[nH\\]','(?<!\\[)n','o']
    lowerlist = re.findall(r"(?=("+'|'.join(string_lst)+r"))",bead_smi)
    
    #Construct test rings for aromatic fragments
    if lowerlist:
        frag_size = len(lowerlist)
        #For two atoms + substituents, make a 3-membered ring
        if frag_size == 2:
            subs = bead_smi.split(''.join(lowerlist))
            for i in range(len(subs)):
                #Rare error where carbonyl output =O rather then O=
                if 'O=' in subs[i]:
                    subs[i]=subs[i][::-1]
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
    
def get_types(beads,mol,ring_beads,matched_maps):
    #loops through beads and determines bead type
    script_path = os.path.dirname(os.path.realpath(__file__))
    DG_data = read_DG_data('{}/fragments-exp.dat'.format(script_path))

    #Use DASH to calculate bead partial charges
    if args.q: 
        #command = ["python3","./DASH_Charges.py"]
        #subprocess.run(command)
        print("Accessing DASH to access partial charge - Note if there are crashes, hydrogens need to be indicated in the SMILES, e.g. CC[N+](C)([H])C")
        Bead_Partial_Charges=Partial_Charges(smi,list(beads))
        for i in range(len(Bead_Partial_Charges)): Bead_Partial_Charges[i]=round(Bead_Partial_Charges[i],2)

    bead_types = []
    charges = []
    all_smi = []
    h_donor,h_acceptor = get_hbonding(mol,beads)
    for i,bead in enumerate(beads):
        qbead = sum([mol.GetAtomWithIdx(int(j)).GetFormalCharge() for j in bead])
        charges.append(qbead)
        bead_smi,ring_size, frag_size = get_smi(bead,mol)
        all_smi.append(bead_smi)
        Bead_Partial_Charge=0
        if args.q: 
            Bead_Partial_Charge=Bead_Partial_Charges[i]
        bead_types.append(param_bead(beads,bead,bead_smi,ring_size,frag_size,any(i in ring for ring in ring_beads),qbead,i in h_donor,i in h_acceptor,DG_data,matched_maps,Bead_Partial_Charge))
    for i in all_smi:
        if 'F' in i:
            bead_types=reassign_halogen_adjacent_charge_beads(bead_types,beads,all_smi)
            break
    
    #Activate neighbour group tuning. Not integrated with charged beads!
    if args.t:
        bead_types = tune_model(beads,bead_types,all_smi)
    
    split_charge=False
    count=0
    if args.q: 
        for i in bead_types:
            if 'q' in i:
                split_charge=True
                break
        if split_charge:
            for i in bead_types: 
                if 'q'in i:
                    count += 1
            splitcharge=1/count
            for z,i in enumerate(bead_types): 
                if charges[z]!=0 and round(splitcharge,3)*count !=1:
                    charges[z]=splitcharge+(1-round(splitcharge,3)*count)
                elif 'q' in i:
                    charges[z]=splitcharge
                else:charges[z]=0
    return bead_types,charges,all_smi,DG_data

def reassign_halogen_adjacent_charge_beads(bead_types,beads,all_smi):
    #If >2 F atoms are in the bead adjacent to certain charge beads, change the assignment
        
    # Code to apply alternative halogenated charge bead assignments. Only verified for PFAS surfactants.
    for z,btype in enumerate(bead_types):
        if 'Q' in btype:
    
            scores,ties = rank_nodes(A_cg)
            tuned = []
            fixed = []
            for rank in ties:
                for node in rank:
                    if not any(node in ring for ring in ring_beads):
                    
                        bonded = [j for j in np.nonzero(A_cg[node])[0]]
                        for nbor in bonded:
                            temp=all_smi[nbor]
                            #Check if more then 2 fluroines in the adjacent bead
                            if 'F' in temp and node==z:
                                temp=temp.replace('F','',1)
                                if 'F' in temp:
                                    if 'O=[SH](=O)[O-]' in all_smi[z]:
                                        bead_types[z]='Q1p'
                                    elif 'O=C[O-]' in all_smi[z]:
                                        bead_types[z]='SQ1p'
    return bead_types

def tune_model(beads,bead_types,all_smi):
    #Gets pairs of beads for tuning by ordering beads by centrality and grouping bonded pairs 
    
    scores,ties = rank_nodes(A_cg)
    original=bead_types.copy()

    def is_tunable(nbor):
        #Don't tune beads if part of ring, predefined fragment, or only contains C
        if any(nbor in ring for ring in ring_beads):
            return False
        elif not any(element in all_smi[nbor] for element in ['O','N','S','F','Cl','Br','I']):
            return False
        #elif any(charge in all_smi[nbor] for charge in ['+','-']):
            #return False
        else:
            return True

    tuned = []
    fixed = []

    for rank in ties:
        for bead in rank:
            if not any(bead in ring for ring in ring_beads):
                bonded = [j for j in np.nonzero(A_cg[bead])[0]]
                for nbor in bonded:
                    if scores[nbor] >= scores[bead] and is_tunable(nbor):
                        tuned.append(nbor)
                        fixed.append(bead)
                if (bead not in fixed) and is_tunable(bead) and len(bonded) >= 1:
                    tuned.append(bead)    
                    fixed.append(bonded[0])

    for t,f in zip(tuned,fixed):
        bead_types[t] = tune_bead(beads[t],bead_types[t],beads[f],bead_types[f])

    #Code to ensure Q beads are unchanged in the process of tuning: log kow a poor comparison for ions
    for i,x in enumerate(original):
        if 'Q' in x:
            bead_types[i]=original[i]

    return bead_types
#
#def tuning_pairs(all_smi):
#
#    not_tuned = []
#    tuned = []
#
#    for i,bsmi in enumerate(all_smi):
#        if any(i in ring for ring in ring_beads):
#            not_tuned.append(i) #Don't tune ring beads
#        elif not any(element in bsmi for element in ['O','N','S','F','Cl','Br','I']:
#            not_tuned.append(i) #Don't tune alkyl chains
#        else:
#            tuned.append(i) #Potentially tune everything else
#
#    #Find suitable reference beads (if any)
#    for t in tuned:
#        bonded = [j for j in np.nonzero(A_cg[t])[0]]
#        for n in bonded[:]:
#            if any(n in ring for ring in ring_beads):
#                

def get_diffs(alogps,ring_size,frag_size,category,size):
    #Gets free energy differences between fragment and all bead types
    diffs = np.abs(np.array(delta_Gs[ring_size-frag_size][category][size]) - alogps)

    return diffs

def param_bead(beads,bead,bead_smi,ring_size,frag_size,ring,qbead,don,acc,DG_data,matched_maps,Bead_Partial_Charge):
    #Parametrises bead type
    #types = ['P6','P5','P4','P3','P2','P1','N6','N5','N4','N3','N2','N1','C6','C5','C4','C3','C2','C1']

    #Check for SMARTS matches
    btype = ''
    for m,match in enumerate(matched_maps):
        if sorted(match) == sorted(bead):
            #Overwrite Q assignement if Q < 0.75
            if args.q:
                if qbead != 0 and 0.75 <=  abs(Bead_Partial_Charge):
                    btype = matched_beads[m]
            else:
                btype = matched_beads[m]
            
    #Check if preset beads are available - these are suggested bead types. Not forced. Check for anagrams
    presets=list(preset_beads.items())
    preset=[]
    for pair in presets:
        preset.append(pair[0])
    check=False
    #Check if character is preset in any beads 
    for single in preset:
        current=single
        if len(bead_smi)==len(single):
            for char in bead_smi:
                if char in single:
                    single=single.replace(char,"",1)
                    if single=="" and char==bead_smi[-1]:
                        btype = preset_beads[current]
                    continue
                else:
                    break
            #btype = preset_beads[bead_smi]

    #if btype == '':
        #Get h-bonding label
    #if don and not acc:
    #    category = 'da'
    #    suffix = 'd'
    #elif acc and not don:
    #    category = 'da'
    #    suffix = 'a'
    #else:
    category = 'standard'
    suffix = ''
    #if halogenated, use halogenated beads (X)
    #if 'Cl' in bead_smi or 'F' in bead_smi or 'Br' in bead_smi or 'I' in bead_smi:
    count=0
    for char in bead_smi:
        if 'F' in char and ring==False:
            count=count+1
    if count>=2:category='halogen'

    types = m3_beads[category]

    path_length = get_size(bead,path_matrix)#path_length counts bonds spanning fragment
        
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

    if btype == '':
        #Parametrise charged beads based on h-bonding behaviour
        if (qbead!=0 and not args.q) or (qbead != 0 and 0.75 <=  abs(Bead_Partial_Charge) and args.q):
            btype = 'Qx' #placeholder, not a real bead type
        else:
            if 0.25 <= abs(Bead_Partial_Charge):
                suffix = 'q'
                bead_smi=bead_smi.replace("+","")
                bead_smi=bead_smi.replace("-","")
                #bead_smi=bead_smi.replace("[","")
                #bead_smi=bead_smi.replace("]","")
            try:
                #Get from list of precalculated fragments
                print("Lookup ",bead_smi)
                alogps = DG_data[bead_smi]['DG']
            except:
                #If not on list, get from server or Wildmann-Crippen
                print('{} not on list'.format(bead_smi))
                alogps = get_alogps(bead_smi)

            #Get difference between fragment DG_OW and all beads
            diffs = get_diffs(alogps,ring_size,frag_size,category,size)
            sort_diffs = np.argsort(diffs)
            btype = types[sort_diffs[0]]

        btype = prefix + btype + suffix
    
    return btype                        


def get_alogps(bead_smi):
    #Gets ALOGPS value from server. If this fails for whatver reason, use Wildmann-Crippen
    if args.v:print("Generating bead log Kow Values: ")
    try:
        if args.v:print("Accessing ALOGPS Webserver....: ")
        alogps = requests.get('http://vcclab.org/web/alogps/calc?SMILES=' + bead_smi).text
    except:
        if args.v:print("ALOGPS Server access failed")
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        print(bead_smi,'Data from Wildmann-Crippen - i.e. Generated from atomic contributions')
        return logK*5.74
    if 'error' not in alogps:
        if args.v:print("ALOGPS Server access successful")
        logK = float(alogps.split()[4])
    else:
        bead_smi=bead_smi.replace('H','')
        bead_smi=bead_smi.replace('2','')
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        if args.v:print("ALOGPS Server access failed")
        print(bead_smi,'Data from Wildmann-Crippen - i.e. Generated from atomic contributions')
    
    return logK*5.74

def bead_coords(bead,conf):
    #Get coordinates of a bead

    coords = np.array([0.0,0.0,0.0])
    total = 0.0
    
    #Note COM does not include hydrogens in determination, COG does
    if args.COM:
        for atom in bead:
            mass = mol.GetAtomWithIdx(atom).GetMass() 
            coords += conf.GetAtomPosition(atom)*mass
            total += mass
    #COG default
    else:
        for atom in bead:
            coords += conf.GetAtomPosition(atom)
            total+=1
            check=mol.GetAtomWithIdx(atom)
            #Include H (Connectivity not included in beads matrix)
            for hydrogen in check.GetNeighbors():
                symbol=hydrogen.GetSymbol()
                if symbol == 'H':
                    coords += conf.GetAtomPosition(hydrogen.GetIdx())
                    total +=1
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
        

def get_masses(all_smi,A_cg,virtual):
    #Calculate mass of each fragment in amu
    m_H = 1.00727645209
    masses = []
    for b,smi in enumerate(all_smi):
        aa_frag = Chem.MolFromSmiles(smi)
        #mass is mass of whole fragment minus m_H*neighbours
        frag_mass = rdMolDescriptors.CalcExactMolWt(aa_frag)
        excess_mass = np.sum(A_cg[b])*m_H
        masses.append(frag_mass-excess_mass)
    
    
    if args.v:print("Fragment Masses: ",masses)

    #Redistribute virtual masses
    for vsite,refs in virtual.items():
        vmass = masses[vsite]
        masses[vsite] = 0.0
        for rsite,weight in refs.items():
            masses[rsite] += weight*vmass

    if args.v:print("Masses: post redistribution of virtual site mass",masses)
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

    masses = get_masses(all_smi,A_cg,virtual)

    itp.write('\n[atoms]\n')
    
    for b in range(len(bead_types)):
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
                    angles.append([x,int(shared[0]),z])

    #Calculate and write to file
    if angles:
        itp.write('\n[angles]\n')
        coords = np.zeros((len(beads),3))
        thetas = np.zeros(len(angles))
        for conf in mol.GetConformers():
            for i,bead in enumerate(beads):
                coords[i] = bead_coords(bead,conf)
            for a,angle in enumerate(angles):
                vec1 = np.subtract(coords[angle[0]],coords[angle[1]])
                vec1 = vec1/np.linalg.norm(vec1)
                vec2 = np.subtract(coords[angle[2]],coords[angle[1]])
                vec2 = vec2/np.linalg.norm(vec2)
                theta = np.arccos(np.dot(vec1,vec2))
                thetas[a] += theta

        thetas = thetas*180.0/(np.pi*nconfs)


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

def get_coords(mol,beads):
    #Calculates coordinates for output gro file
    mol_Hs = Chem.AddHs(mol)
    conf = mol_Hs.GetConformer(0)

    cg_coords = []

    if args.COM:
        for bead in beads:
            coord = np.array([0.0,0.0,0.0])
            total = 0.0
            for atom in bead:
                mass = mol.GetAtomWithIdx(atom).GetMass()
                coord += conf.GetAtomPosition(atom)*mass
                total += mass
            coord /= (total*10.0)
            cg_coords.append(coord)


    else:
        #COG Default
        for bead in beads:
            coord = np.array([0.0,0.0,0.0])
            total = 0.0
            for atom in bead:
                coord += conf.GetAtomPosition(atom)
                total += 1
                
                #Include hydrogen in cg coordinate assignment
                check=mol_Hs.GetAtomWithIdx(atom)
                for hydrogen in check.GetNeighbors():
                    symbol=hydrogen.GetSymbol()
                    if symbol == 'H':
                        #print("Hydrogen: ",hydrogen, conf.GetAtomPosition(hydrogen.GetIdx()))
                        coord += conf.GetAtomPosition(hydrogen.GetIdx())
                        total += 1
            coord /= (total*10)
            cg_coords.append(coord)

    cg_coords_a = np.array(cg_coords)
    return cg_coords_a

def get_smarts_matches(mol):
    #Get matches to SMARTS strings
    smarts_strings = {
    'S([O-])(=O)(=O)O'  :    'Q2',
    '[S;!$(*OC)]([O-])(=O)(=O)'   :    'Q3',# Reparameterised to Q3 from SQ4. Q1p for polyfluorinated
    'C[N+]([C])([C])[C]' : 'SQ3',  #Tetramethylammonium, Benzyl quat. Brached and hard to access hence assignment
    '[C;D2][N+;D4]([C;D2])([C;D2])[C;D2]' : 'Q4', #Tetraalkylammonium, highly branched.
    '[N+;D2][C;D1]' : 'TQ2', # Pyridinium, Imideazolium and reserve cases for small highly branched fragments.
    '[C][N+;D3]([C;D1])[C;D1]' : 'Q1p', # Tertiary Ammonium
    'CC[N+;D2]C' : 'SQ1p', #Secondary Ammonium
    'C[N+;D2]C(C)C' : 'Q3', #Secondary Ammonium
    'CCC[N+]' : 'SQ1p',  #Primary Ammonium
    #'CCC[N+;D1]' : 'Q4',  #Primary Ammonium
    #'C[N+;D2]' : 'TQ4',  #Primary Ammonium
    'O=C[O;D2]':'SP2',    #Parameterisations for esters from diester paper
    'O=C[O-;D1]' : 'SQ5n', # SQ1p for polyfluorinated, SQ5n normally
    #'C[O]C' : 'TN6', # Parameterisations for >= 4 ether units. Default SN4
    '[N+](=O)[O-]' : 'SN3a', # Parameterisation from Martini 3 small molecules paper (https://doi.org/10.1002/adts.202100391). Off by default, causes an error in O=[N+]([O-])c1ccc(-c2nc3cc4nc5ccccc5nc4cc3[nH]2)cc1O. Under investiation 
    'CC[N+](C)(C)[O-]' : 'P6',
    'CP(=S)(C)[S-]' : 'Q1'
    #'CC' : 'C2',
    #'OO' : 'P5'
    #'CCC' : 'C2',
    #'CCCC': 'C2'
    }

    matched_maps = []
    matched_beads = []
    
    already_matched=[]
    
    #if >=4 contiguous ethers in molcule
    if "[O-]S(=O)(=O)OCCOCCOCCOCCOCC" in args.s or "CCOCCOCCOCCOCCOS(=O)(=O)[O-]" in args.s:
        smarts_strings['[C;R0][C;R0][O;D2]']='TN6'
    elif "CCOCCOCCOCCO" in args.s:
        smarts_strings['[C;R0][O;D2][C;R0]']='TN6'

    for smarts in smarts_strings:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        #Generate temp adjacency matrix
        A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol),dtype='f')
        if matches: 
            print("")
            print("Comparing Hard-Coded Matches: ", matches)
            print("")
        
        #Check if there are any overlapping matches: start from end of list which has minimum overlaps. Include all matched_maps in comparison
        for count,match in enumerate(matches):
            if len(matches)>1:
                temp_lists=already_matched
                
                res=[]
                for z in matches:
                    res.append([value for value in match if value in z])
                    res=[x for x in res if x]
                #Check if mapping is at end or start of molecules. Only reorder if not
                if len(res[-1]) < len(res[0]) and max(res[0]) != len(A_atom[:,0])-1 and max(res[-1]) != len(A_atom[:,0])-1 and min(res[0])!=0 and min(res[-1])!=0:
                    matches=list(matches)
                    matches=list(reversed(matches))
                    matches[count]=list(reversed(matches[count]))
                    break

        for match in matches:            

            #For dealing with duplicated matches, i.e. CCOC or COCC
            if match in already_matched:continue
            repeat=False
            for l in match:
                for k in already_matched:
                    for j in k:
                        if l in k:
                            repeat=True
            if repeat==True:
                continue

            #Check matched row for bonded groups. Discard those within the matched group and identify if lone atoms are created
            #PROBLEM: Compares all matched fragments. i.e. in C14EO4S multiple ether mappings are rejected, not just the final one
            bonded_groups=[]
            already_matched.append(match)
            lone=False            

            for row in match:
                for i,element in enumerate(A_atom[row]):   #Take matched map row and identify other atoms the atom is bonded to (element in row)
                    if element!=0 and i not in match and i not in already_matched:
                        #For atoms adjactent to match atoms, identify if it has any other bonds
                        count=0
                        for x,adj in enumerate(A_atom[i]):
                            #Subloop to check if any adjacent atom is in a manually mapped group, to prevent clashes. Unnecssary?
                            #But should allow two adjacent groups i.e.[O-]C(=O)C[N+](C)(C)C
                            matched=False
                            for y in already_matched:
                                if x in y:  #If the adjacent atom is already_matched
                                    #Needs to identify whether any aleady mapped atoms are in the new bead
                                    #The fact that they are already matched is not a problem
                                    matched=True
                                    
                            #if adj!=0 and matched==False:
                            #If not adjacent to anything
                            if adj!=0:
                                count+=1
                        if count ==0:
                            print("Matched Map ",match,", ",smarts,"leaves lone atom: Number",i, "Discarding")
                            lone=True
                            continue                        

            #If no lone atom is created, allow mapping. On first iteration, loop will allow mappings that might prevent furhter mappings of neighbouring functional groups, be aware!
            if lone==False:
                if args.v: print("Hard Coded fragement recognised: ", smarts)
                matched_maps.append(list(match))
                matched_beads.append(smarts_strings[smarts])
                already_matched.append(match)

    return matched_maps,matched_beads


def tune_bead(var_bead,var_type,fix_bead,fix_type):
    dimer_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,fix_bead+var_bead)

    dimer_DG = get_alogps(dimer_smi)

    var_size = var_type[0] if (var_type[0] in ['T','S']) else 'R'
    fix_size = fix_type[0] if (fix_type[0] in ['T','S']) else 'R'

    var_cat = 'standard'
    fix_cat = 'standard'

    fix_base = fix_type[1:] if fix_type[1].isalpha() else fix_type

   # # Testing. Treat Q bead as P for purpose of tuning
   # if 'Q' in fix_base:
   #     fix_base=fix_base.replace('Q','P')
   #     if fix_base[-1]=='n' or fix_base[-1]=='p':
   #         fix_base=fix_base[:-1]

    #Get closest sum of two beads
    fix_DG = delta_Gs[0][fix_cat][fix_size][m3_beads[fix_cat].index(fix_base)]
    dimer_sum = np.asarray(delta_Gs[0][var_cat][var_size]) + fix_DG
    dimer_diff = np.abs(dimer_sum - dimer_DG)
    var_base = m3_beads[var_cat][np.argmin(dimer_diff)]

    var_type = var_base if (var_size == 'R') else (var_size+var_base)

    return var_type

def Partial_Charges(smi,mapping):
    from serenityff.charge.tree.dash_tree import DASHTree
    
    # Load the default tree
    tree = DASHTree()
    
    #test versions
    #smiles='[O-]C1=CC=C(Cl)C=C1'
    #mapping=[[5, 3, 4], [0, 1, 2], [6, 7]] 

    molH = Chem.AddHs(Chem.MolFromSmiles(smi))

    #Get all-atom adjacency matrix
    A_atomH = np.asarray(Chem.GetAdjacencyMatrix(molH),dtype='f')
    
    #Assumes for indexes that heavy atoms print first, then hydrogens
    last_heavy_atom=max(max(mapping))
    charges = tree.get_molecules_partial_charges(molH)["charges"]  #Lookup partial charges
    
    #for i in range(len(charges)):
    #    print(i,charges[i])

    bead_charges=[]
    for bead in mapping:
        charge_sum=0
        for atom in bead:
            charge_sum=charge_sum+charges[int(atom)]

            for row,i in enumerate(A_atomH[:, atom]):
                if i !=0 and row>last_heavy_atom:
                    charge_sum=charge_sum+charges[row]
        bead_charges.append(charge_sum)
    return(bead_charges)

def molecule_image(mol,name):

    #Output 2D image of molecule with atom indexes labeled: Atom 0 unlabelled
    canvas_width_pixels = 500
    canvas_height_pixels  = 500
   
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    mol_draw = rdMolDraw2D.PrepareMolForDrawing(mol)

    #mol_draw = Chem.RemoveHs(mol)
    for atom in mol_draw.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    
    drawer = rdMolDraw2D.MolDraw2DSVG(canvas_width_pixels,canvas_height_pixels)
    drawer.DrawMolecule(mol_draw)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open('output'+name+'.svg', 'w') as f:
        f.write(svg)

#Parse System Arguments and Provide useful outputs. Code starts below
parser = argparse.ArgumentParser(description='Script to generate a coarse grained itp and gro files from a SMILES code')
parser.add_argument('-s',help='SMILES code of the molecule.',required=True)
parser.add_argument('-f',help='Name of molecule: name for output gro and itp files',required=True)
parser.add_argument('-v',help='Verbose Mode: Output useful print statements throughout mapping and parameterisation.',action='store_true')
parser.add_argument('-t',help='Tuning: Setting to enable tuned bead parameterisation. This uses the log Kow of neighbouring beads as well as the log Kow of the bead in question when parameterising a bead. Developed focusing on diesters, for an upcoming publication.',action='store_true')
parser.add_argument('-COM',help='Option to reuse old COM mapping of beads. COG mapping default, no option required. COM not advised by Martini Developers!',action='store_true')
parser.add_argument('-p',help='Have RDKIT output an index-labeled image of your molecule.',action='store_true')
parser.add_argument('-q',help='Assess Partial Charges of beads. If >0.25 or <0.75 replace Q beads with neutral beads with the q flags. Useful for charged rings',action='store_true')
args = parser.parse_args()

#Start of script introduction
print("")
print("Thanks for using cg_kmw. Latest version: https://github.com/cgkmw-durham/cg_param_m3")
print("")
print("The original version of this script for the Martini 2 forcefield, and a full description of the mapping and") 
print("parametrisation procedures, can be found in the following paper:")
print("T.D. Potter, E.L. Barrett and M.A. Miller, Automated Coarse-Grained Mapping Algorithm for the Martini Force Field and") 
print("Benchmarks for Membrane–Water Partitioning, J. Chem. Theory Comput., 2021, https://doi.org/10.1021/acs.jctc.1c00322.")
print("")
if args.v: print("Colour coding of dumped arrays: ", "\033[38;5;34m","Atoms ","\033[0;0m", "Vs ", "\033[38;5;128m","Beads","\033[0;0m")
if args.v: print("Atoms Numbered According to smiles, bead mapping is arbitrary")

#Generate molecule object
smi = args.s
mol_name = 'MOL'
mol = Chem.MolFromSmiles(smi)
mol_dict = Chem.MolFromSmiles(smi) #Create second mol object to allow atom mapping; this allows tracking of atoms when assessing molecular fragments
#mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
print("SMILES: ",Chem.MolToSmiles(mol))
if args.p:molecule_image(mol,'_molecule')

#Coarse-grained mapping
print("Performing CG Mapping:")
matched_maps,matched_beads = get_smarts_matches(mol)
ring_atoms = get_ring_atoms(mol)
if args.v: print("Ring Atoms if any: ", "\033[38;5;34m", ring_atoms, "\033[0;0m")
A_cg,beads,ring_beads,path_matrix = mapping(mol,ring_atoms,matched_maps,3,mol_dict)
non_ring = [b for b in range(len(beads)) if not any(b in ring for ring in ring_beads)]

if args.v: print("")
if args.v: print("Final Mapping: ", "\033[38;5;34m",beads,"\033[0;0m")
if args.v: print("")

#Parametrise beads
print("Performing CG Parameterisation:")

if args.t: print("Atom tuning is active: beads will be reassessed based on adjacent groups log Kow, rather then just their own fragments log Kow")
if args.t: print("")

bead_types,charges,all_smi,DG_data = get_types(beads,mol,ring_beads,matched_maps)

if args.v: print("Bead Types: ", bead_types)
if args.v: print("Bead Charges: ", charges)

#Generate atomistic conformers
if args.v: print("")
print("Generating Atomistic Conformers:")
if args.v: print("")
nconfs = 200
mol = Chem.AddHs(mol)
AllChem.EmbedMultipleConfs(mol,numConfs=nconfs,randomSeed=random.randint(1,1000),useRandomCoords=True)
AllChem.UFFOptimizeMoleculeConfs(mol)
coords0 = get_coords(mol,beads)

#Calculate bonded interactions and write gromacs files

write_gro(mol_name,bead_types,coords0,args.f + '.gro')
write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,args.f + '.itp')

if args.v: print("")
print("All done. Thanks for using cg_param!")
