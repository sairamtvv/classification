#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:32:22 2022

@author: sai
"""

import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self,x,y):
        self.x1 = x
        self.y1 = y
        
    def __getitem__(self,index):
        # Get one item from the dataset
        return self.x1[index], self.y1[index]
    
    def __len__(self):
        return len(self.x1)