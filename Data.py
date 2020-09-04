#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:43:02 2019

@author: shuix007
"""

import dgl
import numpy as np
from ase.units import Hartree, eV
import math
import torch
from scipy.spatial import distance
from scipy.sparse import csr_matrix

attr_index = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
atom_index = {'H':0, 'C':1, 'N':2, 'O':3, 'F':4}
atom_type = ['H', 'C', 'N', 'O', 'F']

lg_node_type = []
lg_node_index = {}

for i in range(len(atom_index)):
    for j in range(i, len(atom_index)):
        lg_node_index[(i, j)] = len(lg_node_type)
        lg_node_type.append((i, j))

def my_float(num):
    ''' my function to convert complicated string to float
    '''
    try:
        return float(num)
    except:
        pos = num.find('*^')
        base = num[:pos]
        exp = num[pos+2:]
        return float(base + 'e' + exp)

def setxor(a, b):
    n = len(a)
    
    res = []
    link = []
    i, j = 0, 0
    while i < n and j < n:
        if a[i] == b[j]:
            link.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    
    if i < j:
        res.append(a[-1])
    elif i > j:
        res.append(b[-1])
    else:
        link.append(a[-1])
    
    return res, link

class Molecule(object):
    def __init__(self, filename, cut_r):
        ''' @brief Function to read in molecues
            @params filename: .xyz file
                    cut_r: cut off distance
        '''
        self.cut_r = cut_r
        
        # read in basic information
        f = open(filename)
        self.nattr = len(attr_index)
        self.na = int(f.readline()) # number of atoms
        
        properties = f.readline().split() # list of properties
        self.id = int(properties[1])
        self.properties = dict()
        
        self.atoms = np.zeros(self.na, dtype=np.int64)
        self.coordinates = np.zeros((self.na, 3), dtype=np.float32)
        self.charge = np.zeros(self.na, dtype=np.float32)
        
        for i in range(self.nattr):
            self.properties[attr_index[i]] = float(properties[i + 2])
        
        # convert Hartree to eV
        self.properties['homo'] *= Hartree / eV
        self.properties['lumo'] *= Hartree / eV
        self.properties['gap'] *= Hartree / eV
        self.properties['zpve'] *= Hartree / eV
        self.properties['U0'] *= Hartree / eV
        self.properties['U'] *= Hartree / eV
        self.properties['H'] *= Hartree / eV
        self.properties['G'] *= Hartree / eV

        for i in range(self.na):
            tp = f.readline().split()
            self.atoms[i] = atom_index[tp[0]]
            self.coordinates[i, :] = [my_float(tp[j]) for j in range(1, 4)]
            self.charge[i] = my_float(tp[4])
        
        # skip one line
        tp = f.readline()
        
        # extract the smile representation
        tp = f.readline()
        self.smile = tp[:tp.find('\t')]
        f.close()
        
        self.dist = distance.cdist(self.coordinates, self.coordinates, 'euclidean')
        np.fill_diagonal(self.dist, np.inf)
        
        self._build_hetero_graph()
    
    def _build_hetero_graph(self):
        ################################
        # build the atom to atom graph #
        ################################
        self.dg_num_nodes = self.na
        self.dg_node_feat_discrete = torch.LongTensor(self.atoms)
        
        dist_graph_base = self.dist.copy()
        self.dg_edge_feat = torch.FloatTensor(dist_graph_base[dist_graph_base < self.cut_r]).unsqueeze(1)
        dist_graph_base[dist_graph_base >= self.cut_r] = 0.
        
        atom_graph = dgl.graph(csr_matrix(dist_graph_base), 'atom', 'a2a')
        
        ################################
        # build the bond to bond graph #
        ################################
        num_atoms = self.dist.shape[0]
        bond_feat_discrete = []
        bond_feat_continuous = []
        indices = []
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                a = self.dist[i, j]
                
                if a < self.cut_r:
                    bond_feat_continuous.append([a])
                    indices.append([i, j])
                    tp = tuple(sorted(self.atoms[[i, j]]))
                    bond_feat_discrete.append(lg_node_index[tp])
        
        num_bonds = len(indices)
        self.lg_num_nodes = num_bonds
        self.lg_node_feat_discrete = torch.LongTensor(bond_feat_discrete)
        self.lg_node_feat_continuous = torch.FloatTensor(bond_feat_continuous)
        
        #######################################################
        # build the atom to bond graph and bond to atom graph #
        #######################################################
        assignment = np.zeros((num_atoms, num_bonds), dtype=np.int64)
        for i, idx in enumerate(indices):
            assignment[idx[0], i] = 1
            assignment[idx[1], i] = 1
        
        bipartite_graph_base = csr_matrix(assignment)
        atom2bond_graph = dgl.bipartite(bipartite_graph_base, 'atom', 'a2b', 'bond')
        bond2atom_graph = dgl.bipartite(bipartite_graph_base.transpose(), 'bond', 'b2a', 'atom')
        
        ################################
        # build the bond to bond graph #
        ################################
        bond_graph_base = assignment.T @ assignment
        np.fill_diagonal(bond_graph_base, 0) # eliminate self connections
        bond_graph = dgl.graph(csr_matrix(bond_graph_base), 'bond', 'b2b')
        
        self.hetero_graph = dgl.hetero_from_relations([atom_graph, atom2bond_graph, bond2atom_graph, bond_graph])
        
        ##############################################
        # build edge feature for the bond2bond graph #
        ##############################################
        x, y = np.where(bond_graph_base > 0)
        num_edges = len(x)
        edge_feat_continuous = np.zeros_like(x, dtype=np.float32)

        for i in range(num_edges):
            body1 = indices[x[i]]
            body2 = indices[y[i]]
            
            bodyxor, link = setxor(body1, body2)
            
            a = self.dist[body1[0], body1[1]]
            b = self.dist[body2[0], body2[1]]
            c = self.dist[bodyxor[0], bodyxor[1]]
            
            edge_feat_continuous[i] = self._cos_formula(a, b, c) # calculate the cos value of the angle (-1, 1)
            
        self.lg_edge_feat = torch.FloatTensor(edge_feat_continuous).unsqueeze(1)
    
    def _cos_formula(self, a, b, c):
        ''' formula to calculate the angle between two edges
            a and b are the edge lengths, c is the angle length.
        '''
        res = (a**2 + b**2 - c**2) / (2 * a * b)
        
        # sanity check
        res = -1. if res < -1. else res
        res = 1. if res > 1. else res
        return np.arccos(res)
    
    def get_hetero_graph(self):
        return self.hetero_graph
    
    def get_dg_node_feat_discrete(self):
        return self.dg_node_feat_discrete
    
    def get_lg_node_feat_continuous(self):
        return self.lg_node_feat_continuous
    
    def get_lg_node_feat_discrete(self):
        return self.lg_node_feat_discrete
    
    def get_dg_edge_feat(self):
        return self.dg_edge_feat
    
    def get_lg_edge_feat(self):
        return self.lg_edge_feat
    
    def get_dg_num_nodes(self):
        return self.dg_num_nodes
    
    def get_lg_num_nodes(self):
        return self.lg_num_nodes
