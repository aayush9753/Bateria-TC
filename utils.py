import os
from Bio import SeqIO
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
import json
from itertools import product

cuda = torch.cuda.is_available()
import time

def get_tax_df(path):
    sg_reads_seq = []
    sg_reads_tax = []

    for seq_record in SeqIO.parse(path, "fasta"):
        sg_reads_seq.append(seq_record.seq)
        sg_reads_tax.append(seq_record.description[seq_record.description.find('description=')+12+1:-1].split())
    tax = ["Class", "Order", "Family", "Genus"]
    
    sg = pd.DataFrame()
    for i in range(4):
        col = []
        for j in sg_reads_tax:
            if (len(j))==3:
                j.append('Simiduia')
            col.append(j[i])
        sg[tax[i]] = col

    sg['seq'] = [str(i) for i in sg_reads_seq]
    
    return sg

def clean_tax_df(sg, train):
    tax = {}
    
    classes = list(pd.get_dummies(sg['Class']).columns)
    classes = [i for i in classes]
    
    tax['Class'] = [classes, len(classes)]
    
    orders = []
    for i in sg['Order']:
        if i[0:2]==   r"\"":
            i = i[2:-2]
        orders.append(i)
    sg['Order'] = orders
    orders = list(pd.get_dummies(sg['Order']).columns)
    orders = [ i for i in orders]
    tax['Order'] = [orders, len(orders)]

    families = []
    for i in sg['Family']:
        if i[0:2]==   r"\"":
            i = i[2:-2]
        families.append(i)   
    sg['Family'] = families
    families = list(pd.get_dummies(sg['Family']).columns)
    families = [i for i in families]
    tax['Family'] = [families, len(families)]

    genes = list(pd.get_dummies(sg['Genus']).columns)
    genes = [i for i in genes]
    tax['Genus'] = [genes, len(genes)]
    
    if train:
        labels = []
        for i in range(len(sg)):
            label_i = []

            label_i.append(tax['Class'][0].index(sg['Class'][i])) 
            label_i.append(tax['Order'][0].index(sg['Order'][i])) 
            label_i.append(tax['Family'][0].index(sg['Family'][i])) 
            label_i.append(tax['Genus'][0].index(sg['Genus'][i])) 

            labels.append(label_i)
        sg['labels'] = labels

    return sg, tax

def getKmers(sequence, size = 8):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def preprocessing(path, train = True, size = 8):
    # getting the data
    sg = get_tax_df(path)
    # cleaning the data
    sg_cleaned, tax = clean_tax_df(sg, train = True)
    sg_cleaned['words'] = sg_cleaned.apply(lambda x: getKmers(x['seq'], size=8), axis=1)
    return sg_cleaned, tax


def save_vocabulary(dict_, name): 
    with open("vocab/" + name + ".json", "w") as outfile: 
        json.dump(dict_, outfile)
            
def encoder(sg, min_padding = 10):
    texts = list(sg['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(texts[item])    
    cv = CountVectorizer(ngram_range=(1,1)) # initial : (4, 4)
    cv = cv.fit(texts)

    pad_idx = max(list(cv.vocabulary_.values())) + 1
    word2idx = {}
    word2idx = cv.vocabulary_.copy()
    word2idx['pad'] = pad_idx

    idx2word = {}
    for i in list(word2idx.keys()):
        idx2word[word2idx[i]] = i

    size = max([len(i) for i in sg['words']]) + min_padding

    tokenized_words = []
    for i in sg['words']:
        k = [word2idx[j] for j in i ]
        tokenized_words.append(k + [word2idx['pad']] * (size - len(k)))
    sg['encoded'] = tokenized_words
    
    os.makedirs("vocab", exist_ok=True)
    save_vocabulary(word2idx, 'word2idx')
    save_vocabulary(idx2word, 'idx2word')
    
    return sg

