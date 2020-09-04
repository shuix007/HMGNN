#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 01:01:53 2020

@author: dminerx007
"""

import torch
import torch.nn as nn

def GlorotOrthogonal(tensor, scale=2.):
    nn.init.orthogonal_(tensor)
    tensor.mul_(torch.sqrt(scale / ((tensor.size(0) + tensor.size(1)) * torch.var(tensor, unbiased=False))))