#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:44:43 2019

@author: shuix007
"""
import os
import time
import math
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from Model import HMGNN
from Model.torch_layers import shifted_softplus
from Data import Molecule
from DataLoader import DataLoader
from Util import load_model_state, save_model_state, evaluate, evaluate_gap, lr_scheduler, post_op_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y, model, optimizer):
    model.train()
    batch_size = batch_hg.batch_size
    
    optimizer.zero_grad()
    dg_y_hat, lg_y_hat, y_hat, _ = model(batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat)
    
    dg_loss = F.l1_loss(dg_y_hat.squeeze(), y, reduction='sum')
    lg_loss = F.l1_loss(lg_y_hat.squeeze(), y, reduction='sum')
    cb_loss = F.l1_loss(y_hat.squeeze(), y, reduction='sum')
    
    loss = (dg_loss + lg_loss + cb_loss) / (3 * batch_size)
    if math.isnan(loss.item()) or math.isinf(loss.item()):
        raise RuntimeError('Something is wrong with the Loss.')
    loss.backward()
    optimizer.step()
    post_op_process(model)
    
    return dg_loss.item(), lg_loss.item(), cb_loss.item()

def trainIter(model, 
              optimizer, 
              trn_param, 
              prpty, 
              model_dir, 
              train_data, 
              val_data=None, 
              tst_data=None, 
              batch_size = 32, 
              total_steps = 3000000,
              save_steps = 5000,
              eval_steps = 5000,
              tol_steps = 1000000,
              decrease_steps = 2000000,
              lr_decrease_rate = 0.1):
    
    best_mae = trn_param['best_mae']
    best_iter = trn_param['best_iter']
    iteration = trn_param['iteration']
    log = trn_param['log']
    start = time.time()
    
    dist_graph_loss, line_graph_loss, combined_loss = 0., 0., 0.
    for it in range(iteration + 1, total_steps + 1):
        batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y = trn_data.next_random_batch(batch_size, prpty)
        
        cuda_hg = batch_hg.to(device)
        dg_node_feat_discrete = dg_node_feat_discrete.to(device)
        lg_node_feat_continuous = lg_node_feat_continuous.to(device)
        lg_node_feat_discrete = lg_node_feat_discrete.to(device)
        dg_edge_feat = dg_edge_feat.to(device)
        lg_edge_feat = lg_edge_feat.to(device)
        
        y = y.to(device)
        
        dg_loss, lg_loss, cb_loss = train(cuda_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y, model, optimizer)
        
        dist_graph_loss += dg_loss
        line_graph_loss += lg_loss
        combined_loss += cb_loss
        
        end = time.time()
        
        if it % eval_steps == 0:
            dg_val_mae, lg_val_mae, cb_val_mae, _ = evaluate(model, val_data, prpty, 128, False)
            end_val = time.time()
            
            print('-----------------------------------------------------------------------')
            print('Steps: %d / %d, time: %.4f, val_time: %.4f.' % (it, total_steps, end - start, end_val - end))
            print('Dist graph loss: %.6f, line graph loss: %.6f, combined loss: %.6f.' % (dist_graph_loss / (eval_steps * batch_size), line_graph_loss / (eval_steps * batch_size), combined_loss / (eval_steps * batch_size)))
            print('Val: Dist graph MAE: %.6f, line graph MAE: %.6f, combined MAE: %.6f.' % (dg_val_mae, lg_val_mae, cb_val_mae))
            
            log += '-----------------------------------------------------------------------\n'
            log += 'Steps: %d / %d, time: %.4f, val_time: %.4f. \n' % (it, total_steps, end - start, end_val - end)
            log += 'Dist graph loss: %.6f, line graph loss: %.6f, combined loss: %.6f. \n' % (dist_graph_loss / (eval_steps * batch_size), line_graph_loss / (eval_steps * batch_size), combined_loss / (eval_steps * batch_size))
            log += 'Val: Dist graph MAE: %.6f, line graph MAE: %.6f, combined MAE: %.6f. \n' % (dg_val_mae, lg_val_mae, cb_val_mae)
            
            if cb_val_mae < best_mae:
                best_mae = cb_val_mae
                best_iter = it
                torch.save(model, os.path.join(model_dir, 'Best_model.pt'))
            
            start = time.time()
            dist_graph_loss, line_graph_loss, combined_loss = 0., 0., 0.
            
        if it % decrease_steps == 0:
            optimizer = lr_scheduler(optimizer, lr_decrease_rate)
        
        # stop training if the mae does not decrease in tol_steps on validation set
        if it - best_iter > tol_steps:
            break
        
        if it % save_steps == 0:
            trn_param['iteration'] = it
            trn_param['best_mae'] = best_mae
            trn_param['best_iter'] = best_iter
            trn_param['log'] = log
            save_model_state(model, optimizer, trn_param, os.path.join(model_dir, 'checkpoint.tar'))
            
            # write the log
            f = open(os.path.join(model_dir, 'log.txt'), 'w')
            f.write(log)
            f.close()
    
    # write the log
    log += 'The best iter is %d!, best val MAE is %.6f. \n' % (best_iter, best_mae)
    f = open(os.path.join(model_dir, 'log.txt'), 'w')
    f.write(log)
    f.close()
    
    return best_mae, best_iter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/N_50000_cut_5_seed_1/',
                        help='directory to the data.')
    parser.add_argument('--prpty', type=str, default='U0',
                        help='the property to be trained on.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of elements per batch.')
    parser.add_argument('--train', type=int, default=0,
                        help='validation or model training.')
    parser.add_argument("--num_interaction_residual", type=int, default=1,
                        help="number of residual layers for node output")
    parser.add_argument("--num_atom_residual", type=int, default=1,
                        help="number of residual layers for node output")
    parser.add_argument("--num_convs", type=int, default=5,
                        help="number of convolution layers")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--edge_feat_dim", type=int, default=64,
                        help="dimension of edge features")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--feat_drop", type=float, default=.0,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate.")
    parser.add_argument("--lr_decrease_rate", type=float, default=0.1,
                        help="learning rate decreasing rate.")
    parser.add_argument("--decrease_steps", type=int, default=2000000,
                        help="steps to decrease the learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help="weight decay")
    parser.add_argument('--cut_r', type=float, default=5.,
                        help="cut radius to build graphs")
    parser.add_argument('--return_attn', type=int, default=0,
                        help="return attention scores or not.")
    parser.add_argument('--model_dir', type=str, default='model_foo/',
                        help="working space.")
    args = parser.parse_args()
    
    # training
    if args.train:
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        
        # load training set and validation set
        trn_data = DataLoader(os.path.join(args.data_dir, 'train.data'))
        val_data = DataLoader(os.path.join(args.data_dir, 'val.data'))
        
        # calculate mean and std from the training set
        dg_mean, lg_mean, dg_std, lg_std = trn_data.get_statistics(args.prpty)
        print('Per atom mean: %.7f, std: %.7f.' % (dg_mean, dg_std))
        print('Per edge mean: %.7f, std: %.7f.' % (lg_mean, lg_std))
        
        model = HMGNN(num_convs = args.num_convs,
                      dg_node_type_universe = trn_data.dg_node_type_universe,
                      lg_node_type_universe = trn_data.lg_node_type_universe,
                      dg_num_interaction_residuals = args.num_interaction_residual,
                      lg_num_interaction_residuals = args.num_interaction_residual,
                      dg_num_residuals = args.num_atom_residual,
                      lg_num_residuals = args.num_atom_residual,
                      rbf_dim = int(args.hidden_dim/2),
                      cut_r = args.cut_r,
                      dg_mean = dg_mean,
                      lg_mean = lg_mean,
                      dg_std = dg_std,
                      lg_std = lg_std,
                      hidden_dim = args.hidden_dim,
                      activation = shifted_softplus,
                      feat_drop = args.feat_drop).to(device)
        
        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
        
        # try to load previous training
        try:
            trn_param = load_model_state(model, optimizer, os.path.join(args.model_dir, 'checkpoint.tar'))
        except:
            trn_param = {'iteration':0, 'best_mae': np.inf, 'best_iter': 1, 'log':str(args)+'\n'}
        
        best_mae, best_iter = trainIter(model,
                                        optimizer,
                                        trn_param,
                                        prpty = args.prpty,
                                        model_dir = args.model_dir,
                                        train_data = trn_data,
                                        val_data = val_data,
                                        batch_size = args.batch_size,
                                        decrease_steps = args.decrease_steps,
                                        lr_decrease_rate = args.lr_decrease_rate)
        
    # test
    else:
        print(args)
        
        tst_data = DataLoader(os.path.join(args.data_dir, 'test.data'))
        trn_data = DataLoader(os.path.join(args.data_dir, 'train.data'))
        if args.prpty != 'gap':
            model = torch.load(os.path.join(args.model_dir, 'Best_model.pt')).to(device)
            train_mae_1, train_mae_2, train_mae_all, train_attn = evaluate(model, trn_data, args.prpty, batch_size = args.batch_size, return_attn = args.return_attn)
            test_mae_1, test_mae_2, test_mae_all, test_attn = evaluate(model, tst_data, args.prpty, batch_size = args.batch_size, return_attn = args.return_attn)
            if args.return_attn:
                np.save(os.path.join(args.model_dir, 'train_attn_score_%s.npy' % (args.prpty)), train_attn)
                np.save(os.path.join(args.model_dir, 'test_attn_score_%s.npy' % (args.prpty)), test_attn)
        else:
            model_homo = torch.load(os.path.join(args.model_dir, 'Best_homo_model.pt')).to(device)
            model_lumo = torch.load(os.path.join(args.model_dir, 'Best_lumo_model.pt')).to(device)
            train_mae_1, train_mae_2, train_mae_all = evaluate_gap(model_homo, model_lumo, trn_data, batch_size = args.batch_size)
            test_mae_1, test_mae_2, test_mae_all = evaluate_gap(model_homo, model_lumo, tst_data, batch_size = args.batch_size)
        print('Train: One body MAE: %.6f, two body MAE: %.6f, combined MAE: %.6f.' % (train_mae_1, train_mae_2, train_mae_all))
        print('Test: One body MAE: %.6f, two body MAE: %.6f, combined MAE: %.6f.' % (test_mae_1, test_mae_2, test_mae_all))
