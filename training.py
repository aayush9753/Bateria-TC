from tqdm import tqdm, tqdm_notebook
import numpy as np

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, tqdm_notebook

random_state = 4
from itertools import product
import time

cuda = torch.cuda.is_available()

class Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        X = torch.tensor(self.X[index])
        Y = torch.tensor(self.Y[index])
        return {"X" : X, "Class": Y[0], "Order" : Y[1],
                "Family" : Y[2], "Genus" : Y[3]}

    def __len__(self):
        return len(self.Y)

def train_network(network, train_iter, optimizer, loss_fn, epoch_num, tax):

    epoch_loss = 0 # loss per epoch
    epoch_acc = 0 # accuracy per epoch
    
    network.train() # set the model in training mode as it requires gradients calculation and updtion
    # turn off while testing using  model.eval() and torch.no_grad() block
    
    batch_count = 0
    for batch in tqdm(train_iter,f"Epoch: {epoch_num}"): 
        # data will be shown to model in batches per epoch to calculate gradients per batch
        
        X = batch['X']
        Y = batch[tax]
        if cuda:
            X = X.to('cuda')
            Y = Y.to('cuda')
        optimizer.zero_grad() # clear all the calculated grdients from previous step
        
        predictions = network(X)#.squeeze(1) # squeeze out the extra dimension [batch_size,1]
        
        loss = loss_fn(predictions, Y.long()) # calculate loss on the whole batch
        
        pred_classes = predictions.argmax(axis = 1)
        correct_preds = (pred_classes == Y).float()
        # get a floating tensors of predicted classes  which match original true class 
        
        accuracy = correct_preds.sum()/len(correct_preds)# it'll be a tensor of shape [1,]
        
        # below two are must and should be used only after calculation of Loss by optimizer
        loss.backward() # Start Back Propagation so that model can calculate gradients based on loss
        optimizer.step() # update the weights based on gradient corresponding to each neuron
        
        epoch_loss += loss.item()  # add the loss for this batch to calculate the loss for whole epoch
        epoch_acc += accuracy.item() # .item() tend to give the exact number from the tensor of shape [1,]
        
        
        time.sleep(0.001) # for tqdm progess bar
        if batch_count % 50 == 0:
            print(f"Batch : {batch_count}/{len(train_iter)}   Loss : {loss.item()}   Accuracy : {accuracy.item()}" )
        batch_count += 1
        
    return epoch_loss/len(train_iter), epoch_acc/len(train_iter)

def evaluate_network(network, val_test_iter, optimizer, loss_fn, tax):
    total_loss = 0  # total loss for the whole incoming data
    total_acc = 0 # total accuracy for the whole data
    
    network.eval() # set the model in evaluation mode to not compute gradients and reduce overhead
    
    with torch.no_grad(): # turn of gradients calculation 
        
        for batch in val_test_iter:
            X = batch['X']
            Y = batch[tax]
            if cuda:
                X = X.to('cuda')
                Y = Y.to('cuda')

            predictions = network(X)#.squeeze(1) # squeeze out the extra dimension [batch_size,1]
        
            loss = loss_fn(predictions, Y.long()) # calculate loss on the whole batch
        
            pred_classes = predictions.argmax(axis = 1)
            correct_preds = (pred_classes == Y).float()
            # get a floating tensors of predicted classes  which match original true class 
        
            accuracy = correct_preds.sum()/len(correct_preds)# it'll be a tensor of shape [1,]

            total_loss += loss.item() 
            total_acc += accuracy.item()

        return total_loss/len(val_test_iter), total_acc/len(val_test_iter)