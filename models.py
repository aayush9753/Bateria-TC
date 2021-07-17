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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_state = 4
device
from itertools import product
import time

cuda = torch.cuda.is_available()

class Network(torch.nn.Module):
    def __init__ (self, in_neuron, padding_idx, num_layers = 5, embedding_dim=128, hidden_size=256, out_neuron = 4,
                  m_type='lstm', drop=0.13, bidirectional = True, **kwargs):
        super(Network,self).__init__()
        self.m_type = m_type
        
        self.embedding = torch.nn.Embedding(in_neuron,embedding_dim) # embedding layer is always the first layer
        
        if self.m_type == 'lstm':
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers = num_layers, bidirectional=bidirectional, **kwargs)
        else:
            self.rnn = torch.nn.RNN(embedding_dim, hidden_size, num_layers = num_layers, bidirectional=bidirectional, **kwargs) 
        
        self.dropout = torch.nn.Dropout(drop) # drop the values by random which comes from previous layer
        if bidirectional:
            self.dense = torch.nn.Linear(hidden_size * 2,out_neuron) # last fully connected layer
        else:
            self.dense = torch.nn.Linear(hidden_size, out_neuron)
    
    def forward(self,t):
        
        embedding_t = self.embedding(t) # usually we replace the same tensor as t = self.layer(t)
        drop_emb = self.dropout(embedding_t)
        if self.m_type == 'lstm':
            out, (hidden_state,_) = self.lstm(drop_emb)
        else:
            out, hidden_state = self.rnn(drop_emb)
            
        out = torch.mean(out, axis = 1)
        out = self.dense(out)
        return out