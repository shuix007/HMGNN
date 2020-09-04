#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 00:45:54 2020

@author: dminerx007
"""

import torch
import torch.nn as nn
import numpy as np

class HeteroGraphNorm(nn.Module):
    def __init__(self, hidden_dim):
        super(HeteroGraphNorm, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, heterograph, feat, node_type):
        batch_num_nodes = heterograph.batch_num_nodes(node_type)
        seg = torch.from_numpy(np.array(batch_num_nodes, dtype=np.float32).repeat(batch_num_nodes)).unsqueeze(-1)
        seg = seg.to(feat.device)
        return self.bn(feat / seg.sqrt())