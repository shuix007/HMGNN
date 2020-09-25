#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:45:45 2020

@author: dminerx007
"""

import os
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from Data import Molecule

def parse_id(filename):
    pos = filename.find('.xyz')
    return int(filename[pos-6:pos])

def train_val_test_split(DATADIR, evilmols, train_num, val_num, test_num=None):
    file_list = os.listdir(DATADIR)
    molecular_list = np.array([parse_id(file) for file in file_list])
    molecular_list = np.setdiff1d(molecular_list, evilmols)
    np.random.shuffle(molecular_list)
    train_idx = molecular_list[:train_num]
    val_idx = molecular_list[train_num:(train_num + val_num)]
    if test_num:
        test_idx = molecular_list[(train_num + val_num):(train_num + val_num + test_num)]
    else:
        test_idx = molecular_list[(train_num + val_num):]
    
    return train_idx, val_idx, test_idx

def load_badmoleculars(filename):
    evilmols = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines[9:-1]:
            evilmols.append(int(line.split()[0]))
    return np.array(evilmols)

def preprocess(DATADIR, indices, cut_r):
    data = []
    for idx in tqdm(indices):
        filename = DATADIR + 'dsgdb9nsd_' + str(idx).zfill(6) + '.xyz'
        mol = Molecule(filename, cut_r)
        data.append(mol)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATADIR', type=str, default='../QM9/qm9/',
                        help='directory to the data.')
    parser.add_argument('--target_dir', type=str, default='data_0',
                        help='directory to save the data.')
    parser.add_argument('--evil_filename', type=str, default='../QM9/uncharacterized.txt',
                        help='filename of evil molecules.')
    parser.add_argument('--split_filename', type=str, default='No',
                        help='filename of evil molecules.')
    parser.add_argument('--train_num', type=int, default=110000,
                        help='number of training instances.')
    parser.add_argument('--val_num', type=int, default=10000,
                        help='number of validation instances.')
    parser.add_argument('--test_num', type=int, default=10831,
                        help='number of test instances.')
    parser.add_argument('--save_mol_index', type=int, default=1,
                        help='Save index of train, val, test.')
    parser.add_argument('--cut_r', type=float, default=5.,
                        help='cut off distance.')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed.')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if not os.path.isdir('Data'):
        os.mkdir('Data')
    
    if not os.path.isdir(os.path.join('Data', args.target_dir)):
        os.mkdir(os.path.join('Data', args.target_dir))
        
    if args.split_filename != 'No':
        split = np.load(args.split_filename)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']
        print('Loaded splits.')
    else:
        evilmols = load_badmoleculars(args.evil_filename)
        train_idx, val_idx, test_idx = train_val_test_split(args.DATADIR, evilmols, args.train_num, args.val_num, args.test_num)
        if args.save_mol_index:
            np.savez(os.path.join('Data', args.target_dir, 'split_train_%d.npz' % (args.train_num + args.val_num)), 
                     train_idx=train_idx, 
                     val_idx=val_idx, 
                     test_idx=test_idx)
        print('Failed to load splits, generating new split.')
    
    data = preprocess(args.DATADIR, train_idx, args.cut_r)
    pickle.dump(data, open(os.path.join('Data', args.target_dir, 'train.data'), 'wb'))
    del data
    
    data = preprocess(args.DATADIR, val_idx, args.cut_r)
    pickle.dump(data, open(os.path.join('Data', args.target_dir, 'val.data'), 'wb'))
    del data
    
    data = preprocess(args.DATADIR, test_idx, args.cut_r)
    pickle.dump(data, open(os.path.join('Data', args.target_dir, 'test.data'), 'wb'))
    del data
