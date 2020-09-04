#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:30:40 2020

@author: dminerx007
"""

import dgl
import torch
import pickle
import numpy as np
from ase.units import Hartree, eV

from Data import Molecule, lg_node_type

dgl.load_backend('pytorch')

attr_index = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
atom_index = {'H':0, 'C':1, 'N':2, 'O':3, 'F':4}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(object):
    def __init__(self, filename):
        ''' Load data
        '''
        self.data = pickle.load(open(filename, 'rb'))
        self.dg_node_type_universe = len(atom_index)
        self.lg_node_type_universe = len(lg_node_type)
        
        # build a reference for atoms, 2 bodies and 3 bodies
        self.dg_node_ref = np.zeros(len(self.data) + 1, dtype=np.int64)
        for i in range(len(self.data)):
            self.dg_node_ref[i+1] = self.dg_node_ref[i] + self.data[i].dg_num_nodes
            
        self.lg_node_ref = np.zeros(len(self.data) + 1, dtype=np.int64)
        for i in range(len(self.data)):
            self.lg_node_ref[i+1] = self.lg_node_ref[i] + self.data[i].lg_num_nodes
        
        self.trn_iter = 0
        self.val_iter = 0
        self.indices = np.arange(len(self.data), dtype=np.int64)
        
        self.ref = dict()
        self.ref['zpve'] = torch.FloatTensor([0., 0., 0., 0., 0.]) * Hartree / eV
        self.ref['U0'] = torch.FloatTensor([-0.500273, -37.846772, -54.583861, -75.064579, -99.718730]) * Hartree / eV
        self.ref['U'] = torch.FloatTensor([-0.498857, -37.845355, -54.582445, -75.063163, -99.717314]) * Hartree / eV
        self.ref['H'] = torch.FloatTensor([-0.497912, -37.844411, -54.581501, -75.062219, -99.716370]) * Hartree / eV
        self.ref['G'] = torch.FloatTensor([-0.510927, -37.861317, -54.598897, -75.079532, -99.733544]) * Hartree / eV
        self.ref['Cv'] = torch.FloatTensor([2.981, 2.981, 2.981, 2.981, 2.981])
        
        # initialize yhat
        self.yhat = dict()
        self.y = dict()
        for prpty in attr_index:
            if prpty in self.ref:
                yhat = torch.FloatTensor([self.ref[prpty][mol.atoms].sum().item() for mol in self.data]) 
            else:
                yhat = torch.zeros(len(self.data), dtype=torch.float32)
            y = torch.FloatTensor([mol.properties[prpty] for mol in self.data])
            self.yhat[prpty] = [yhat]
            self.y[prpty] = y
        
    def __len__(self):
        return len(self.data)
    
    def get_statistics(self, prpty):
        dg_mean, dg_count, lg_mean, lg_count = 0., 0., 0., 0.
        
        dg_per_diff = torch.zeros(self.dg_node_ref[-1], dtype=torch.float32)
        lg_per_diff = torch.zeros(self.lg_node_ref[-1], dtype=torch.float32)
        
        for i, m in enumerate(self.data):
            target_value = self.y[prpty][i] - self.yhat[prpty][0][i]
            dg_per_diff[self.dg_node_ref[i]:self.dg_node_ref[i+1]] = target_value / m.dg_num_nodes
            lg_per_diff[self.lg_node_ref[i]:self.lg_node_ref[i+1]] = target_value / m.lg_num_nodes
            
            dg_mean += target_value
            lg_mean += target_value
            dg_count += m.dg_num_nodes
            lg_count += m.lg_num_nodes
        
        dg_mean = (dg_mean / dg_count).item()
        lg_mean = (lg_mean / lg_count).item()
        dg_std = torch.sqrt(torch.mean((dg_per_diff - dg_mean)**2)).item()
        lg_std = torch.sqrt(torch.mean((lg_per_diff - lg_mean)**2)).item()
        
        return dg_mean, lg_mean, dg_std, lg_std
        
    def next_batch(self, batch_size, prpty):
        ''' generate a sequential batch
        '''
        # when the remaining moleculars is larger than the batch_size
        if self.val_iter + batch_size < len(self.data):
            indices = np.arange(self.val_iter, self.val_iter+batch_size, dtype=np.int64)
            self.val_iter += batch_size
        else:
            indices = np.arange(self.val_iter, len(self.data), dtype=np.int64)
            self.val_iter = 0
        
        return self._generate_batch(indices, prpty)
        
    def next_random_batch(self, batch_size, prpty):
        ''' generate a random batch
        '''
        if self.trn_iter + batch_size < len(self.data):
            indices = self.indices[self.trn_iter:(self.trn_iter + batch_size)]
            self.trn_iter += batch_size
        else:
            np.random.shuffle(self.indices)
            indices = self.indices[:batch_size]
            self.trn_iter = batch_size
        
        return self._generate_batch(indices, prpty)
    
    def _generate_batch(self, indices, prpty):
        batch_g = dgl.batch_hetero([self.data[i].get_hetero_graph() for i in indices])
        
        dg_node_feat_discrete = torch.cat([self.data[i].get_dg_node_feat_discrete() for i in indices], dim=0)
        lg_node_feat_continuous = torch.cat([self.data[i].get_lg_node_feat_continuous() for i in indices], dim=0)
        lg_node_feat_discrete = torch.cat([self.data[i].get_lg_node_feat_discrete() for i in indices], dim=0)
        dg_edge_feat = torch.cat([self.data[i].get_dg_edge_feat() for i in indices], dim=0)
        lg_edge_feat = torch.cat([self.data[i].get_lg_edge_feat() for i in indices], dim=0)
        
        y = self.y[prpty][indices] - self.yhat[prpty][0][indices]
        
        return batch_g, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y
