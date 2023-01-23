#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:56:01 2022

@author: sai
"""
import torch.nn as nn


# Now let's build the  network
class Model(nn.Module):
    def __init__(self,Fashion_ecommerce,input_features):
        super(Model, self).__init__()
        self.Fashion_ecommerce=Fashion_ecommerce
        self.fc1 = nn.Linear(input_features, 15)
        self.fc2 = nn.Linear(15, 8)
        self.fc3 = nn.Linear(8, 6)
        self.fc4 = nn.Linear(6, 3)
        self.fc5 = nn.Linear(3, 1)
        self.droput = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = self.fc5(out)
        out = self.sigmoid(out)
        return out