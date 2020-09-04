#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:09:22 2019

@author: dminerx007
"""

import torch.nn as nn
from .initializer import GlorotOrthogonal
from .normalization import HeteroGraphNorm

class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super(DenseLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        GlorotOrthogonal(self.fc.weight.data)
        if self.bias:
            self.fc.bias.data.zero_()
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))

class ResLayer(nn.Module):
    def __init__(self, node_type, in_dim, hidden_dim, out_dim, activation):
        super(ResLayer, self).__init__()
        self.node_type = node_type
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        
        self.gn1 = HeteroGraphNorm(hidden_dim)
        self.gn2 = HeteroGraphNorm(out_dim)
        
        if in_dim != out_dim:
            self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        GlorotOrthogonal(self.fc1.weight.data)
        GlorotOrthogonal(self.fc2.weight.data)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        
        if self.in_dim != self.out_dim:
            GlorotOrthogonal(self.res_fc.weight)
        
    def forward(self, batch_g, feat):
        identity = feat
        
        out = self.fc1(feat)
        out = self.gn1(batch_g, out, self.node_type)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.gn1(batch_g, out, self.node_type)
        out = self.activation(out)
        
        if self.in_dim == self.out_dim:
            out += identity
        else:
            out += self.res_fc(identity)
        
        return out