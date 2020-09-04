#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:44:23 2020

@author: shuix007
"""

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model_state(model, optimizer, trn_param, filename):
    torch.save({
            'trn_param': trn_param,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, filename)
    
def load_model_state(model, optimizer, filename):
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['trn_param']

def lr_scheduler(optimizer, decrease_rate=0.9):
    """Decay learning rate by a factor."""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decrease_rate

    return optimizer

def post_op_process(model):
    model.dg_input_module.edge_rbf.mu.data.clamp_(0)
    model.dg_input_module.edge_rbf.beta.data.clamp_(0)
    model.lg_input_module.node_rbf.mu.data.clamp_(0)
    model.lg_input_module.node_rbf.beta.data.clamp_(0)
    model.lg_input_module.edge_rbf.mu.data.clamp_(0)
    model.lg_input_module.edge_rbf.beta.data.clamp_(0)
    model.lg_output_module.std.data.clamp_(0)
    model.dg_output_module.std.data.clamp_(0)

def evaluate(model, data, prpty, batch_size = 32, return_attn = False):
    model.eval()
    with torch.no_grad():
        n_batch = len(data) // batch_size
        n_batch = (n_batch + 1) if (len(data) % batch_size) != 0 else n_batch
        dg_MAE = 0.
        lg_MAE = 0.
        cb_MAE = 0.
        attn_list = []
        
        for i in range(n_batch):
            batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y = data.next_batch(batch_size, prpty)
            
            dg_node_feat_discrete = dg_node_feat_discrete.to(device)
            lg_node_feat_continuous = lg_node_feat_continuous.to(device)
            lg_node_feat_discrete = lg_node_feat_discrete.to(device)
            dg_edge_feat = dg_edge_feat.to(device)
            lg_edge_feat = lg_edge_feat.to(device)
            
            y = y.to(device)
            
            dg_y_hat, lg_y_hat, y_hat, attn_score = model(batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat)
            
            # get the prediction score
            dg_MAE += F.l1_loss(dg_y_hat.squeeze(), y, reduction='sum').item()
            lg_MAE += F.l1_loss(lg_y_hat.squeeze(), y, reduction='sum').item()
            cb_MAE += F.l1_loss(y_hat, y, reduction='sum').item()
            
            # get the attention score for each body
            if return_attn:
                attn_list.append(attn_score.detach())
        
        if return_attn:
            attn_list = torch.cat(attn_list, dim=0).cpu().numpy()
            
        return dg_MAE / len(data), lg_MAE / len(data), cb_MAE / len(data), attn_list
    
def evaluate_gap(model_homo, model_lumo, data, batch_size = 32):
    model_homo.eval()
    model_lumo.eval()
    with torch.no_grad():
        n_batch = len(data) // batch_size
        n_batch = (n_batch + 1) if (len(data) % batch_size) != 0 else n_batch
        dg_MAE = 0.
        lg_MAE = 0.
        cb_MAE = 0.
        
        for i in range(n_batch):
            batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y = data.next_batch(batch_size, 'gap')
            
            dg_node_feat_discrete = dg_node_feat_discrete.to(device)
            lg_node_feat_continuous = lg_node_feat_continuous.to(device)
            lg_node_feat_discrete = lg_node_feat_discrete.to(device)
            dg_edge_feat = dg_edge_feat.to(device)
            lg_edge_feat = lg_edge_feat.to(device)
            
            y = y.to(device)
            
            dg_y_homo, lg_y_homo, y_homo, _ = model_homo(batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat)
            dg_y_lumo, lg_y_lumo, y_lumo, _ = model_lumo(batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat)
            
            # get the prediction score
            dg_MAE += F.l1_loss((dg_y_lumo - dg_y_homo).squeeze(), y, reduction='sum').item()
            lg_MAE += F.l1_loss((lg_y_lumo - lg_y_homo).squeeze(), y, reduction='sum').item()
            cb_MAE += F.l1_loss(y_lumo - y_homo, y, reduction='sum').item()
            
        return dg_MAE / len(data), lg_MAE / len(data), cb_MAE / len(data)