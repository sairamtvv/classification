#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:32:22 2022

@author: sai
"""


from dataloader import DataLoader
from model import Model
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

class Fashion_ecommerce():
    """Fashion_ecommerce Class
    Limitations:
        Data imbalnces are not completed considered
        Hyper parameters are not tuned 
        Unittest
        build train and tes  in a single function  
    """

    def __init__(self):
        
        self.dataloader=DataLoader(self)
        
        #All hyperparameters
        #data loader
        self.test_size=0.2  #test size during train test split used in data loader
        self.batch_size=32  #batch size during training 
        
        #training
        self.num_epochs = 50
        
 
    

    def build_train_test(self,filepath):
        """ Builds trains and tests the pytorch model  """
        # Create the network (an object of the Net class)
        self.dataloader.preprocess_data(filepath) #to generate inputfeatures variable
        self.net = Model(self,self.dataloader.inputfeatures)
        train_load, test_load=self.dataloader.preprocess_data(filepath)
        #In Binary Cross Entropy: the input and output should have the same shape 
        #size_average = True --> the losses are averaged over observations for each minibatch
        criterion = torch.nn.BCELoss(size_average = True)   
        #criterion = torch.nn.CrossEntropyLoss()
        # We will use SGD with momentum with a learning rate of 0.1
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.05, momentum=0.9)    


        #Training the FFN
        num_epochs = self.num_epochs

        #Define the lists to store the results of loss and accuracy
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        #Training
        for epoch in range(num_epochs): 
            #Reset these below variables to 0 at the begining of every epoch
            correct = 0
            iterations = 0
            iter_loss = 0.0
            
            self.net.train()                   # Put the network into training mode
            
            for i, (inputs, labels) in enumerate(train_load):
                
                # Convert torch tensor to Variable
                inputs =Variable(inputs)
                labels = Variable(labels)
                labels=labels.unsqueeze(1)
               
                optimizer.zero_grad()            # Clear off the gradient in (w = w - gradient)
                outputs = self.net(inputs)   
                
                loss = criterion(outputs, labels)  
                iter_loss += loss.item()      # Accumulate the loss
                loss.backward()                 # Backpropagation 
                optimizer.step()                # Update the weights
                
                # Record the correct predictions for training data 
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1
            
            # Record the training loss
            train_loss.append(iter_loss/iterations)
            # Record the training accuracy
            train_accuracy.append((100 * correct / self.dataloader.len_train_datset))
           
            #Testing
            loss = 0.0
            correct = 0
            iterations = 0

            self.net.eval()                    # Put the network into evaluation mode
            
            for i, (inputs, labels) in enumerate(test_load):
                
                # Convert torch tensor to Variable
                inputs =Variable(inputs)
                labels = Variable(labels)
                labels=labels.unsqueeze(1)
                
                
                
                outputs = self.net(inputs)
                
                loss = criterion(outputs, labels) # Calculate the loss
                loss += loss.item()
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                
                iterations += 1

            # Record the Testing loss
            test_loss.append(loss/iterations)
            # Record the Testing accuracy
            test_accuracy.append((100 * correct / self.dataloader.len_test_datset))
            
            print ('Epoch {}/{}, Training Loss: {:.3f}'
                   .format(epoch+1, num_epochs, train_loss[-1],))
      
        f = plt.figure(figsize=(10, 10))


        plt.plot(train_loss, label='Training Loss')
        #plt.plot(test_loss, label='Testing Loss')
        plt.legend()
        plt.show()


        


        classes = ['flop', 'top']

        self.net.eval() 
        with torch.no_grad():                   # Put the network into evaluation mode
            y_pred = self.net(self.dataloader.X_test)

        y_pred = y_pred.ge(.5).view(-1).cpu()
        y_test = self.dataloader.y_test.cpu()

        print(classification_report(y_test, y_pred, target_names=classes))


        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)

        hmap = sns.heatmap(df_cm, annot=True, fmt="d")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    def is_it_hit(self,filepath):
        eval_load=self.dataloader.preprocess_data_eval(filepath)
    
        
        self.net.eval()
        with torch.no_grad():
            self.y_pred= []
            for i, (inputs, labels) in enumerate(eval_load):
                
                # Convert torch tensor to Variable
                inputs =Variable(inputs)
                labels = Variable(labels)
                labels=labels.unsqueeze(1)
                
                
                
                outputs = self.net(inputs)
                
                output = (outputs>0.5).float()
                self.y_pred.append(output.item())
                
        df = pd.read_csv(filepath)
        df["success_indicator"]=self.y_pred            
        df['success_indicator'].replace(0, 'Flop',inplace=True)
        df['success_indicator'].replace(1, 'Hit',inplace=True)  
        df.to_csv("updated_with_model.csv", encoding='utf-8', index=False) 
        
        
  
     
if __name__ == '__main__':
    fashion_ecommerce=Fashion_ecommerce()
    fashion_ecommerce.build_train_test('historic.csv')
    fashion_ecommerce.is_it_hit("prediction_input.csv")
    #fashion_ecommerce.is_it_hit("historic_without_output.csv")