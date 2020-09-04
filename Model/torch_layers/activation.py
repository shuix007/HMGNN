#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:41:14 2020

@author: dminerx007
"""

import math
import torch.nn.functional as F

def shifted_softplus(data):
    return F.softplus(data) - math.log(2)