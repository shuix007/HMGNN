#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:42:05 2020

@author: dminerx007
"""
import math
import torch
import torch.nn as nn

class ShrinkDistRBF(nn.Module):
    def __init__(self, K, cut_r, requires_grad=False):
        super(ShrinkDistRBF, self).__init__()
        self.K = K
        self.cut_r = cut_r
        
        # initialize mu and beta in Eq (7)
        self.mu = nn.Parameter(torch.linspace(math.exp(-cut_r), 1., K).unsqueeze(0), requires_grad=requires_grad)
        self.beta = nn.Parameter(torch.full((1, K), math.pow((2 / K) * (1 - math.exp(-cut_r)), -2)), requires_grad=requires_grad)
    
    def forward(self, r):
        batch_size = r.size(0)
        K = self.K
        
        ratio_r = r / self.cut_r
        phi = 1 - 6 * ratio_r.pow(5) + 15 * ratio_r.pow(4) - 10 * ratio_r.pow(3)
        
        phi = phi.expand(batch_size, K)
        local_r = r.expand(batch_size, K)
        
        g = phi * torch.exp(-self.beta.expand(batch_size, K) * (torch.exp(-local_r) - self.mu.expand(batch_size, K))**2)
        
        return g
        
class DistRBF(nn.Module):
    def __init__(self, K, cut_r, requires_grad=False):
        super(DistRBF, self).__init__()
        self.K = K
        self.cut_r = cut_r
        
        # initialize mu and beta in Eq (7)
        self.mu = nn.Parameter(torch.linspace(math.exp(-cut_r), 1., K).unsqueeze(0), requires_grad=requires_grad)
        self.beta = nn.Parameter(torch.full((1, K), math.pow((2 / K) * (1 - math.exp(-cut_r)), -2)), requires_grad=requires_grad)
    
    def forward(self, r):
        batch_size = r.size(0)
        K = self.K

        local_r = r.expand(batch_size, K)
        
        g = torch.exp(-self.beta.expand(batch_size, K) * (torch.exp(-local_r) - self.mu.expand(batch_size, K))**2)
        
        return g

class AngleRBF(nn.Module):
    def __init__(self, K, requires_grad=False):
        super(AngleRBF, self).__init__()
        self.K = K
        
        # initialize mu and beta in Eq (7)
        self.mu = nn.Parameter(torch.linspace(0., math.pi, K).unsqueeze(0), requires_grad=requires_grad)
        self.beta = nn.Parameter(torch.full((1, K), math.pow((2 / K) * math.pi, -2)), requires_grad=requires_grad)
    
    def forward(self, r):
        batch_size = r.size(0)
        K = self.K

        local_r = r.expand(batch_size, K)
        g = torch.exp(-self.beta.expand(batch_size, K) * (local_r - self.mu.expand(batch_size, K))**2)
        
        return g
    
class RBF(nn.Module):
    def __init__(self, K, m, M, requires_grad=False):
        super(RBF, self).__init__()
        self.K = K
        
        # initialize mu and beta in Eq (7)
        self.mu = nn.Parameter(torch.linspace(math.exp(-M), math.exp(-m), K).unsqueeze(0), requires_grad=requires_grad)
        self.beta = nn.Parameter(torch.full((1, K), math.pow((2 / K) * (math.exp(-m) - math.exp(-M)), -2)), requires_grad=requires_grad)
    
    def forward(self, r):
        batch_size = r.size(0)
        K = self.K

        local_r = r.expand(batch_size, K)
        
        g = torch.exp(-self.beta.expand(batch_size, K) * (torch.exp(-local_r) - self.mu.expand(batch_size, K))**2)
        
        return g