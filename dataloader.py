#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:18:58 2022

@author: sai
"""

"""Data Loader"""
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


import torch
from dataset import Dataset

class DataLoader():
    """Data Loader class"""
    def __init__(self,Fashion_ecommerce):
        self.Fashion_ecommerce=Fashion_ecommerce
        
    
    def load_data(self,filepath):
        """Loads dataset from path"""
        data = pd.read_csv(filepath)
        #errorhandling to be written
        sns.countplot(data.success_indicator)
        print(data.success_indicator.value_counts() / data.shape[0])
        print("Need to consider the imbalance in the data set\n")
        #Since, small data passing by value
        return data

    
    def load_data_eval(self,filepath):
        """Loads dataset from path"""
        data = pd.read_csv(filepath)
        
        return data
   
    
   

    def enconding_normalize_data(self,filepath):
        data=self.load_data(filepath)
        # For x: Extract out the dataset from all the rows (all samples) and all columns except last column (all features). 
        # For y: Extract out the last column (which is the label)
        # Convert both to numpy using the .values method
        self.mean_stars=data["stars"].mean()
        self.std_stars=data["stars"].std()
        
        x = data.iloc[:,1:-1]
        print(x)
        
        # Lets have a look some samples from our data
        print(x[:3])
         #normalize
        x["stars"]=(x["stars"]-self.mean_stars)/self.std_stars


        self.ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough')
        self.fit=self.ct.fit(x)
        x=self.fit.transform(x).toarray()
        
        #x = self.ct.transform(x) 
        
        return x
    
    
    
    def enconding_normalize_data_eval(self,filepath):
        data=self.load_data_eval(filepath)
        
        x = data.iloc[:,1:]
        print(x)
        
        # Lets have a look some samples from our data
        print(x[:3])
         #normalize
        x["stars"]=(x["stars"]-self.mean_stars)/self.std_stars
        x=self.fit.transform(x).toarray()
        print("-------")
        print(x)
        return x
    
    
    
    def enconding_normalize_target(self,filepath):
        data=self.load_data(filepath)
        # For x: Extract out the dataset from all the rows (all samples) and all columns except last column (all features). 
        # For y: Extract out the last column (which is the label)
        # Convert both to numpy using the .values method
        y_string= list(data.iloc[:,-1])
        print(y_string[:3])

        # Our neural network only understand numbers! So convert the string to labels
        y_int = []
        for string in y_string:
            if string == 'top':
                y_int.append(1)
            else:
                y_int.append(0)

        # Now convert to an array
        y = np.array(y_int, dtype = 'float64')
        print(y)
        return y
    
    
    
    
    def preprocess_data(self,filepath):
        """ Preprocess and splits into training and test"""
        x=self.enconding_normalize_data(filepath)
        y=self.enconding_normalize_target(filepath)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.Fashion_ecommerce.test_size, stratify=y, random_state=324)
        
        self.inputfeatures=X_train.shape[1]
        
        #print(X_train)

        
        
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        self.X_test=X_test
        self.y_test=y_test
        
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        
        
            
        train_dataset = Dataset(X_train,y_train)
        test_dataset =  Dataset(X_test,y_test)
        self.len_train_datset=len(train_dataset)
        self.len_test_datset=len(test_dataset)
        #Load the data to your dataloader for batch processing and shuffling


        
        #epochs = 10
        #Make the dataset iterable
        train_load = torch.utils.data.DataLoader(dataset = train_dataset, 
                                                 batch_size = self.Fashion_ecommerce.batch_size,
                                                 shuffle = True)

        test_load = torch.utils.data.DataLoader(dataset = test_dataset, 
                                                 batch_size = self.Fashion_ecommerce.batch_size,
                                                 shuffle = False)




        # Let's have a look at the data loader
        print("There is {} batches in the dataset".format(len(train_load)))
        for (x,y) in train_dataset:
            print("For one iteration (batch), there is:")
            print("Data:    {}".format(x.shape))
            print("Labels:  {}".format(y.shape))
            break

        return train_load, test_load

    def preprocess_data_eval(self,filepath):
        """ Preprocess for dataloader evaluation"""
        x=self.enconding_normalize_data_eval(filepath)
        
        
        X = torch.from_numpy(x).float()
        Y_dummy=torch.zeros(X.shape[0])

        print(X.shape, Y_dummy.shape)
        
        
        
            
        
        eval_dataset =  Dataset(X,Y_dummy)
        self.len_eval_dataset=len(eval_dataset)
        
        #Load the data to your dataloader for batch processing and shuffling


        
        #epochs = 10
        #Make the dataset iterable
        

        eval_load = torch.utils.data.DataLoader(dataset = eval_dataset, 
                                                 batch_size = 1,
                                                 shuffle = False)




        # Let's have a look at the data loader
        print("There is {} batches in the dataset".format(len(eval_load)))
        for (x,y) in eval_dataset:
            print("For one iteration (batch), there is:")
            print("Data:    {}".format(x.shape))
            print("Labels:  {}".format(y.shape))
            break

        return eval_load
