"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import glob
import pickle
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random
import numpy as np
import pickle
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import dataloader 
from model import Attention, GatedAttention
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.utils.data as data_utils
import torch
import glob
import pandas as pd




class rsna(Dataset):
    def __init__(self,list_of_file_paths,labels_list):
        self.list_of_file_paths = list_of_file_paths
        self.labels_list = labels_list

    def __getitem__(self,index):

        inputs = self.list_of_file_paths[index]
        infile = open(inputs,'rb')
        inputs = pickle.load(infile)
        infile.close()
        inputs = (inputs-inputs.min())/(inputs.max()-inputs.min())
        
        x = torch.tensor(inputs).float()
        y = self.labels_list[index]
        return x,y

    def __len__(self):
        return len(self.labels_list)

def train_test_load(seed):
    files_path_list = []
    test_inputs_list = []
    test_labels_list=[]
    train_inputs_list =[]
    train_labels_list =[]
    df = pd.read_csv('train_labels.csv')
    labels_list_wo5 = df['MGMT_value'].to_list()

    for id,j in enumerate(sorted(glob.glob('/home/ravi/RSNA_WHOLE/MIL_INPUT_256/*'))):
        files_path_list.append(j)
    print('seed',seed)
    test_indices = Rand(0,len(labels_list_wo5),int(0.1*len(labels_list_wo5)),seed)

    for k in range(len(labels_list_wo5)):
        if k not in test_indices:
            train_inputs_list.append(files_path_list[k])
            train_labels_list.append(labels_list_wo5[k])
        elif k in test_indices:
            test_inputs_list.append(files_path_list[k])
            test_labels_list.append(labels_list_wo5[k])

    return train_inputs_list, train_labels_list, test_inputs_list,test_labels_list

def Rand(start, end, num, seed):
    res = []
    random.seed(seed)
    for j in range(num):
        res.append(random.randint(start, end))
 
    return res

