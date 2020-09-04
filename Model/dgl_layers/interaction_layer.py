#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:18:53 2020

@author: dminerx007
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from ..torch_layers import ResLayer, DenseLayer, GlorotOrthogonal
        
class HoConv(nn.Module):
    def __init__(self,
                 node_type,
                 hidden_dim,
                 n_interaction_residual,
                 n_atom_residual,
                 activation=F.relu,
                 feat_drop=0.0,
                 residual=True):
        
        super(HoConv, self).__init__()
        self.domestic_node_type = node_type
        self.domestic_edge_type = node_type[0] + '2' + node_type[0]
        self.foreign_node_type = 'bond' if node_type == 'atom' else 'atom'
        self.foreign_edge_type = self.foreign_node_type[0] + '2' + node_type[0]
        
        self.residual = residual
        self.hidden_dim = hidden_dim
        self.feat_drop = nn.Dropout(feat_drop)
                
        # parameters for message construction
        self.G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.fc_src_domestic = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        self.fc_src_foreign = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        self.fc_dst = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        self.fc_node_update = DenseLayer(3 * hidden_dim, hidden_dim, activation, bias=True)
        
        # interaction residual layers
        self.interaction_residual_layer = nn.ModuleList()
        for i in range(n_interaction_residual):
            self.interaction_residual_layer.append(ResLayer(node_type, hidden_dim, hidden_dim, hidden_dim, activation))
        
        # atom residual layers
        self.atom_residual_layer = nn.ModuleList()
        for i in range(n_atom_residual):
            self.atom_residual_layer.append(ResLayer(node_type, hidden_dim, hidden_dim, hidden_dim, activation))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        GlorotOrthogonal(self.G.weight.data)
    
    def forward(self, graph, node_feat_domestic, edge_feat, node_feat_foreign):
        h_domestic = self.feat_drop(node_feat_domestic)
        h_foreign = self.feat_drop(node_feat_foreign)
        eh = self.feat_drop(edge_feat)
        
        # message passing on domestic graph
        eh = self.G(eh)
        src_h = self.fc_src_domestic(h_domestic)
        
        graph.nodes[self.domestic_node_type].data.update({'h': src_h})
        graph.edges[self.domestic_edge_type].data.update({'eh': eh})
        
        graph[self.domestic_edge_type].update_all(message_func=fn.u_mul_e('h', 'eh', 'm'),
                                                  reduce_func=fn.sum(msg='m', out='x'),
                                                  etype=self.domestic_edge_type)
        m_domestic = graph.nodes[self.domestic_node_type].data.pop('x')
        
        # message passing on foreign graph
        src_h = self.fc_src_foreign(h_foreign)
        graph.nodes[self.foreign_node_type].data.update({'h': src_h})
        graph[self.foreign_edge_type].update_all(message_func=fn.copy_src('h', 'm'),
                                                 reduce_func=fn.sum(msg='m', out='x'),
                                                 etype=self.foreign_edge_type)
        
        m_foreign = graph.nodes[self.domestic_node_type].data.pop('x')
        
        # clean up the memories on the graph
        graph.nodes[self.domestic_node_type].data.clear()
        graph.edges[self.domestic_edge_type].data.clear()
        graph.nodes[self.foreign_node_type].data.clear()
        
        # node update
        dst_h = self.fc_dst(h_domestic)
        m = torch.cat([dst_h, m_domestic, m_foreign], dim=1)
        m = self.fc_node_update(m)
        
        # interaction residuals
        for layer in self.interaction_residual_layer:
            m = layer(graph, m)
        
        if self.residual:
            h_domestic = h_domestic + m
        else:
            h_domestic = m
        
        # atom residuals
        for layer in self.atom_residual_layer:
            h_domestic = layer(graph, h_domestic)
        
        return h_domestic
