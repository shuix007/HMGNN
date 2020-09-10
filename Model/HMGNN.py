#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:58:05 2019

@author: shuix007
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import dgl
import dgl.backend as bknd

from .dgl_layers import HoConv
from .torch_layers import ResLayer, DenseLayer, DistRBF, AngleRBF, ShrinkDistRBF, GlorotOrthogonal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sum_hetero_nodes(bg, node_type, feats):
    batch_size = bg.batch_size
    batch_num_nodes = bg.batch_num_nodes(node_type)

    seg_id = torch.from_numpy(np.arange(batch_size, dtype='int64').repeat(batch_num_nodes))
    seg_id = seg_id.to(feats.device)

    return bknd.unsorted_1d_segment_sum(feats, seg_id, batch_size, 0)

class DistGraphInputModule(nn.Module):
    def __init__(self, node_type_universe, edge_continuous_dim, hidden_dim, cut_r, activation):
        super(DistGraphInputModule, self).__init__()
        
        # function to convert discrete node feature (atomic number) to continuous node features
        self.node_embedding_layer = nn.Embedding(node_type_universe, hidden_dim)
        
        self.node_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        
        # radial basis function to convert edge distance to continuous feature
        self.edge_rbf = ShrinkDistRBF(edge_continuous_dim, cut_r)
        
        self.edge_input_layer = DenseLayer(edge_continuous_dim, hidden_dim, activation, bias=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize parameters for nodes
        self.node_embedding_layer.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        
    def forward(self, node_feat_discrete, edge_feat_continuous):
        h = self.node_embedding_layer(node_feat_discrete)
        h = self.node_input_layer(h)
        
        eh = self.edge_rbf(edge_feat_continuous)
        eh = self.edge_input_layer(eh)
        return h, eh

class LineGraphInputModule(nn.Module):
    def __init__(self, node_type_universe, node_continuous_dim, edge_continuous_dim, hidden_dim, cut_r, activation):
        super(LineGraphInputModule, self).__init__()
        
        # function to convert discrete body feature to continuous body features
        self.node_embedding_layer = nn.Embedding(node_type_universe, hidden_dim)
        
        # radial basis function to convert body distance to continuous feature
        self.node_rbf = DistRBF(node_continuous_dim, cut_r)
        
        self.node_input_layer = DenseLayer(hidden_dim + node_continuous_dim, hidden_dim, activation, bias=True)
        
        # radial basis function to convert edge distance to continuous feature
        self.edge_rbf = AngleRBF(edge_continuous_dim)
        
        self.edge_input_layer = DenseLayer(edge_continuous_dim, hidden_dim, activation, bias=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize parameters for nodes
        self.node_embedding_layer.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        
    def forward(self, node_feat_continuous, node_feat_discrete, edge_feat_continuous):
        h_discrete = self.node_embedding_layer(node_feat_discrete)
        h_continuous = self.node_rbf(node_feat_continuous)
        
        h = torch.cat([h_discrete, h_continuous], dim=1)
        h = self.node_input_layer(h)
        
        eh = self.edge_rbf(edge_feat_continuous)
        eh = self.edge_input_layer(eh)
        
        return h, eh

class OutputModule(nn.Module):
    def __init__(self, node_type, node_type_universe, hidden_dim, activation, mean, std):
        super(OutputModule, self).__init__()
        self.domestic_node_type = node_type
        
        # output layer which output the final prediction value (for regression)
        self.node_out_residual = nn.ModuleList()
        self.node_out_layer = nn.Linear(hidden_dim, 1, bias=True)
        
        # scale layer (Important for scaling the output)
        self.node_out_scale = nn.Embedding(node_type_universe, 1)
        self.node_out_bias = nn.Embedding(node_type_universe, 1)
        
        self.mean = nn.Parameter(torch.FloatTensor([mean]))
        self.std = nn.Parameter(torch.FloatTensor([std]))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.node_out_bias.weight.data.zero_()
        self.node_out_scale.weight.data.zero_()
        
        GlorotOrthogonal(self.node_out_layer.weight.data)
        self.node_out_layer.bias.data.zero_()
    
    def forward(self, batch_g, h, node_feat_discrete):
        node_out = self.node_out_layer(h)
        node_scale = self.node_out_scale(node_feat_discrete)
        node_bias = self.node_out_bias(node_feat_discrete)
        node_out = node_scale * node_out + node_bias
        
        node_out = node_out * self.std + self.mean
        
        node_score = sum_hetero_nodes(batch_g, self.domestic_node_type, node_out)
        graph_feat = sum_hetero_nodes(batch_g, self.domestic_node_type, h)
        
        return graph_feat, node_score

class FussionModule(nn.Module):
    def __init__(self, num_orders, hidden_dim, activation, negative_slope=0.2):
        super(FussionModule, self).__init__()
        
        self.lkrelu = nn.LeakyReLU(negative_slope)
        self.batch_norm = nn.BatchNorm1d(num_orders * hidden_dim)
        self.fc_layer = DenseLayer(num_orders * hidden_dim, num_orders * hidden_dim, activation, bias=True)
        self.attn_layer = nn.Linear(num_orders * hidden_dim, num_orders, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.attn_layer.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        
    def forward(self, mol_feat, mol_pred):
        ''' mol_feat: concatenation of readout results from 1, 2, 3, ... bodies, n_batch * (n_input * hidden_dim)
            mol_pred: concatenation of predictions from 1, 2, 3, ... bodies, n_batch * n_input
        '''
        h = self.batch_norm(mol_feat)
        h = self.fc_layer(h)
        h = self.attn_layer(h)
        h = self.lkrelu(h)
        attn_score = F.softmax(h, dim=1)
        output = (attn_score * mol_pred).sum(dim=1)
        return output, attn_score
    
class HMGNN(nn.Module):
    def __init__(self,
                 num_convs,
                 dg_node_type_universe,
                 lg_node_type_universe,
                 dg_num_interaction_residuals,
                 lg_num_interaction_residuals,
                 dg_num_residuals,
                 lg_num_residuals,
                 rbf_dim,
                 cut_r,
                 dg_mean,
                 lg_mean,
                 dg_std,
                 lg_std,
                 hidden_dim,
                 activation,
                 feat_drop):
        
        super(HMGNN, self).__init__()
        self.num_convs = num_convs
        self.activation = activation
        
        self.dg_input_module = DistGraphInputModule(dg_node_type_universe, rbf_dim, hidden_dim, cut_r, activation)
        self.lg_input_module = LineGraphInputModule(lg_node_type_universe, rbf_dim, rbf_dim, hidden_dim, cut_r, activation)
        
        self.dg_interaction_layer = nn.ModuleList()
        self.lg_interaction_layer = nn.ModuleList()
        
        for _ in range(num_convs):
            self.dg_interaction_layer.append(
                    HoConv(
                        'atom',
                        hidden_dim,
                        dg_num_interaction_residuals,
                        dg_num_residuals,
                        activation,
                        feat_drop)
                    )
            
            self.lg_interaction_layer.append(
                    HoConv(
                        'bond',
                        hidden_dim,
                        lg_num_interaction_residuals,
                        lg_num_residuals,
                        activation,
                        feat_drop)
                    )
            
        self.dg_output_module = OutputModule('atom', dg_node_type_universe, hidden_dim, activation, dg_mean, dg_std)
        self.lg_output_module = OutputModule('bond', lg_node_type_universe, hidden_dim, activation, lg_mean, lg_std)
            
        self.fussion_layer = FussionModule(2, hidden_dim, activation)
        
    def forward(self, batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat):
        dg_h, dg_eh = self.dg_input_module(dg_node_feat_discrete, dg_edge_feat)
        lg_h, lg_eh = self.lg_input_module(lg_node_feat_continuous, lg_node_feat_discrete, lg_edge_feat)

        for i in range(self.num_convs):
            # message passing layers
            dg_h_new = self.dg_interaction_layer[i](batch_hg, dg_h, dg_eh, lg_h)
            lg_h = self.lg_interaction_layer[i](batch_hg, lg_h, lg_eh, dg_h)
            
            dg_h = dg_h_new

        dg_graph_feat, dg_node_pred = self.dg_output_module(batch_hg, dg_h, dg_node_feat_discrete)
        lg_graph_feat, lg_node_pred = self.lg_output_module(batch_hg, lg_h, lg_node_feat_discrete)
        
        graph_feat = torch.cat([dg_graph_feat, lg_graph_feat], dim=1)
        score = torch.cat([dg_node_pred, lg_node_pred], dim=1)
        
        pred, attn_score = self.fussion_layer(graph_feat, score)
        
        return dg_node_pred, lg_node_pred, pred, attn_score
