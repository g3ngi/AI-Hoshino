pip install --upgrade torch torch-geometric

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from google.colab import files
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from torch.nn import Module
from torch_geometric.nn import GCNConv, global_add_pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda

class Label_Index:
    def __init__(self, dataset):
        self.labels = dataset['type'].unique()
        self.label_index = {label: index for index, label in enumerate(self.labels)}
        self.index_label = {index: label for index, label in enumerate(self.labels)}

    def indexes_labels(self, dataset):
        return dataset['type'].map(self.index_label)

    def labels_indexes(self, dataset):
        return dataset['type'].map(self.label_index)
    def __call__(self, label):
        return self.label_index[label]

label_index = Label_Index(dataset)
label_index('phishing')

class Char_Index:
    def __init__(self, urls) -> None:
        self.char_index = {}
        self.index_char = {}
        for url in urls:
            for char in url:
                if char not in self.char_index:
                    self.char_index[char] = len(self.char_index)
                    self.index_char[len(self.index_char)] = char

    def string_indexes(self, string):
        return [self.char_index[char] for char in string]

char_index = Char_Index(dataset['url'])
char_index.string_indexes(dataset.url[0]), len(char_index.char_index)

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(GRU, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        if self.bidirectional == True:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        if self.bidirectional == True:
            out = out[:, -1, :self.hidden_size] + out[:, 0, self.hidden_size:]
        return self.fc(out)
        gru_model = GRU(len(char_index.char_index), 128, 128, len(label_index.labels), bidirectional=True, num_layers=1)
        gru_model(inputs)

# shuffle data
dataset = dataset.sample(frac=1).reset_index(drop=True)

# split data into train and test
train_data = dataset[:int(len(dataset)*0.8)]
test_data = dataset[int(len(dataset)*0.8):].reset_index(drop=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, char_index: Char_Index, label_index: Label_Index) -> None:
        self.df = df
        self.char_index = char_index
        self.label_index = label_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        url = self.df.url[index]
        label = self.label_index(self.df.type[index])
        return torch.tensor(self.char_index.string_indexes(url)), torch.tensor(label)

trainDataset = Dataset(train_data, char_index, label_index)
testDataset = Dataset(test_data, char_index, label_index)
len(trainDataset), len(testDataset)

!pip3 install discord

files.upload()
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d sid321axn/malicious-urls-dataset
!mkdir dataset
!unzip malicious-urls-dataset.zip -d dataset

EMOJI = '''
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⣤⣤⣤⣄⣀⠀⠀⠉⠣⠙⣳⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⣴⡰⠁⣼⠿⢶⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⠶⡀⣤⢺⣿⡯⠶⠧⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠴⠶⠶⠒⠲⠤⠶⢤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢤⣈⢛⣤⣾⡷⠶⠶⠶⠧⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣶⣾⣍⠀⠀⠀⠀⠀⠀⢠⡆⣧⣄⣉⡛⠛⠿⣿⣿⣿⣿⣿⣿⡿⠿⢴⣒⣛⣥⡆⣠⠀⠀⠀⠀⠀⠀⠈⣶⣴⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡧⡞⣳⣶⢶⡶⢰⣯⢸⢳⡘⣿⣿⣿⣿⡏⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣤⣳⢡⢿⢈⣷⠀⣴⣧⣀⡀⢿⣿⣿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠙⣗⣧⣋⡇⠘⢛⣿⣿⠞⢸⣷⡘⣿⣿⡿⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢳⣿⣿⣦⣿⣶⣿⣷⢾⣹⣶⣾⠝⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣠⠞⠀⠀⠀⣹⠃⠀⠀⠀⠘⠛⠀⢀⣿⣿⣿⣾⣿⣧⣦⣬⣿⣿⣿⣿⣻⣿⣿⣿⣿⣷⣿⣿⣿⠘⢿⣿⣿⡎⢿⣏⠻⣍⠀⠀⠈⢦⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⡼⠃⠀⠀⠀⠀⠀⠀⠀⣽⢹⣿⣿⣿⡏⢿⣯⣿⠋⠙⠙⠟⠿⣿⣽⡏⣿⣿⡏⣿⣿⡌⣆⠙⢿⣷⠈⣏⢦⣸⣦⠀⠀⠀⠹⣄⠀⠀⠀⠀⠀
⠀⠀⠀⢠⠞⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⣸⠃⣸⣿⣿⣿⣷⡘⠛⠉⠀⠀⠀⠀⠀⠙⠋⣸⣿⣿⣿⣿⣿⡇⢿⡄⠀⢹⣀⣸⣿⡇⠙⢧⠀⠀⠀⠈⢣⡀⠀⠀⠀
⠀⠀⡴⠃⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⣰⡇⡸⣿⣿⣿⣿⣿⡻⣆⠀⠀⠀⠀⠀⠀⠀⡰⢿⣿⣿⣿⣿⣿⣧⠈⣇⠀⠸⠋⠙⢿⣿⣦⡀⢳⠀⠀⠀⠀⠙⣄⠀⠀
⠀⠼⠁⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⢰⢻⢧⢣⣿⣿⣿⣿⣿⣿⣮⣀⠈⠉⠒⠺⠗⣨⣶⣿⣿⣿⣿⣿⣿⣿⠀⠞⡆⠀⠀⠀⠈⣿⣿⡷⡀⢣⠀⠀⠀⠀⠘⣆⠀
⠀⠀⠀⠀⠀⢀⡴⠓⠦⣄⡀⠀⠀⠀⠀⣠⣏⣸⡏⣼⣿⣿⣿⣿⣿⣿⣿⣿⣙⣲⣤⣖⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⡦⣥⣹⠀⠀⠀⢀⣿⣿⡧⠷⠚⢦⡀⠀⠀⠀⠈⣆
⠲⠤⣀⠀⠀⠉⠀⠀⠀⠀⠈⠉⠒⠒⢦⠿⣧⠀⠉⠉⠙⢿⠻⣿⣿⣿⠿⣏⡛⠯⠭⠽⢛⡝⢿⣿⣿⣿⣿⡿⠉⠀⢰⣿⡿⠤⠔⠒⠚⠉⠁⠀⠀⠀⠀⠙⠀⠀⣀⡤⠞
⠀⠀⠈⠑⠢⢄⣀⠀⠀⠀⠀⠀⠀⠀⠘⣆⣼⡆⢀⣀⣠⣼⣶⣿⣿⣿⡦⣵⣍⡓⠒⣚⣭⣮⡾⣽⣿⣿⣿⣃⣠⣤⠾⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⠚⠉⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠈⠙⠒⠤⣀⡀⠀⠀⠀⠹⣿⠛⠉⠉⠉⢹⣹⠺⡏⣿⣷⣷⣯⣿⣿⣫⣿⡯⣲⣿⣿⣿⠏⠉⠀⠀⠀⣟⡟⠀⠀⠀⠀⢀⣠⠴⡞⣩⠄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠑⠲⣤⣸⣿⡆⠀⠀⢀⣼⡟⢓⡿⢼⣿⣳⡙⠿⡿⠟⣡⠞⣹⣿⣿⡟⠻⣄⠤⠤⣤⣿⣄⣠⠤⣶⣾⣿⣿⣾⣞⣋⡴⠃⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⡏⠀⠀⠤⠶⠿⣧⠼⠇⠀⣿⣇⠻⣶⠬⠘⢁⣞⣿⣿⡿⣶⡞⠁⠀⠀⠀⣼⣿⡌⢣⡹⣿⣿⣿⣭⡭⢟⡁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⣽⣿⣿⡀⢀⣠⡤⠄⢹⡀⢀⡄⣿⣿⠛⠒⣖⠋⢿⣼⣿⣿⠃⠺⣷⣄⠀⠀⠚⠿⣿⣿⡄⠳⣄⢻⠉⠉⠛⠋⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡟⣱⣿⣿⣿⡿⠛⠉⠀⣀⢹⢷⣾⣾⣿⣿⠸⡟⠉⣻⠎⣿⣿⣿⠀⠀⠀⠙⢧⣄⢀⣼⣿⡏⢿⡄⠹⣮⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠋⢠⣿⣿⣿⣿⣷⣀⡤⠞⢁⣈⣷⣿⣿⣿⣿⠀⢳⠟⡟⢠⣿⣿⣿⣆⠀⠀⠀⠈⢻⣿⣿⣿⣿⠘⣷⠀⠹⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⡞⢹⠀⣾⢻⣿⣿⣿⣿⣿⠀⢠⣝⣻⣿⣿⣿⣿⡏⠀⠈⣾⡄⢸⣿⣿⣿⣿⣿⣶⠶⣾⣿⣿⣿⣿⣽⠀⢹⡇⠀⢸⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⠏⠀⢸⣸⡏⢸⣿⣿⣿⣿⣿⣿⣾⣭⠿⠿⠟⣻⡿⠀⠀⠀⣷⣧⠈⣿⡻⢿⢿⢿⣿⠟⣿⣿⣿⣿⣿⢸⡄⠈⣷⠀⣸⠙⣧⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢠⡟⠀⠀⣼⣿⠁⡿⣏⣿⣿⣿⣿⣿⣿⣷⣤⣤⣌⣡⣇⣀⠀⠀⣿⣿⣀⣈⣳⣼⣟⣋⡿⢰⣿⣿⣿⣿⡟⢸⠁⠀⣿⢠⡏⠀⢹⡆⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠸⡇⠀⢰⠁⣿⡀⢻⠹⣿⣿⡜⣿⣿⣿⣧⢻⡳⡶⣴⠥⢿⡿⠿⣿⣿⣿⣧⠼⣿⠉⣽⡀⣿⣿⣿⡟⣼⠁⣾⠀⠀⣿⠏⠀⠀⢘⡇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠻⡀⠈⠀⢹⠻⢾⡀⠈⠻⣷⡈⢿⣿⡏⠀⠳⣟⡙⣇⠸⣷⠀⢸⣿⠏⠀⣠⡏⢰⣿⠀⣹⣿⠟⣴⠃⣰⠇⠀⢠⡿⠀⠀⠀⠘⠃⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠇⠀⠀⠈⠛⠂⠈⠓⢦⣶⣽⣷⣜⣀⣛⣇⣼⣿⣀⣼⣯⣴⣿⣿⡿⠟⣡⠞⠁⠴⠋⠀⠀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡾⠀⠸⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢟⠟⠀⢷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠃⠀⠀⠙⢦⡉⠉⠙⠛⠛⠛⠛⠛⠉⠉⢀⡴⠋⠀⣴⠀⠓⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠴⠋⠀⠀⣀⣤⡀⠈⠀⠙⠷⣄⡀⠀⠀⠀⣀⣤⡶⠋⠀⠀⠀⡏⠦⣀⠀⠈⠳⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠊⠁⠀⠀⢀⣾⠿⠋⠀⢀⢖⡴⠀⠈⠁⠀⣴⣋⠽⠋⠠⣄⡠⡀⠰⠁⠀⠈⠳⣦⡀⠀⠙⠤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠞⠁⠀⠀⣰⣷⠋⠀⠀⠀⠀⠀⠉⠁⠀⠀⠀⠈⠻⣾⣦⠀⠀⠐⠒⠤⣽⣦⣀⠀⠀⠀⠉⠉⠒⠤⣀⠀⠀⠀⠀⠀⠀
'''

"""TRAININING MODEL"""


dataset = dataset.sample(frac=1).reset_index(drop=True)

train_data = dataset[:int(len(dataset)*0.8)]
test_data = dataset[int(len(dataset)*0.8):].reset_index(drop=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, char_index: Char_Index, label_index: Label_Index) -> None:
        self.df = df
        self.char_index = char_index
        self.label_index = label_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        url = self.df.url[index]
        label = self.label_index(self.df.type[index])
        return torch.tensor(self.char_index.string_indexes(url)), torch.tensor(label)

trainDataset = Dataset(train_data, char_index, label_index)
testDataset = Dataset(test_data, char_index, label_index)
len(trainDataset), len(testDataset)

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(GRU, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        if self.bidirectional == True:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        if self.bidirectional == True:
            out = out[:, -1, :self.hidden_size] + out[:, 0, self.hidden_size:]
        return self.fc(out)
        gru_model = GRU(len(char_index.char_index), 128, 128, len(label_index.labels), bidirectional=True, num_layers=1)
        gru_model(inputs)

def collate_fn(batch):
  urls, labels = zip(*batch)
  urls = nn.utils.rnn.pad_sequence(urls, batch_first=True)
  return urls, torch.tensor(labels)

trainGenerator = torch.utils.data.DataLoader(trainDataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=2)
testGenerator = torch.utils.data.DataLoader(testDataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=2)

optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()
epochs = 1
torch.cuda.empty_cache()

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, recall_score
from sklearn.preprocessing import label_binarize


train_roc_auc = []
test_roc_auc = []
train_recall = []
test_recall = []

train_losses = []
test_losses = []
train_f1_scores = []
test_f1_scores = []
best_right = 0.0


gru_model.cuda()

for epoch in range(epochs):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    loss_value = 0.0
    gru_model.train()
    predictions = []
    targets = []

    for inputs, label in tqdm(trainGenerator):
        inputs = inputs.cuda()
        label = label.cuda()


        optimizer.zero_grad()
        output = gru_model(inputs)
        l = loss(output, label)
        l.backward()
        loss_value += l.item()
        optimizer.step()


    train_losses.append(loss_value/len(trainGenerator))

    gru_model.eval()
    loss_value = 0.0
    right_num = 0
    all_predictions = []
    all_labels = []

    for inputs, label in testGenerator:
        inputs = inputs.cuda()
        label = label.cuda()
        output = gru_model(inputs)
        l = loss(output, label)
        loss_value += l.item()

        predictions = torch.argmax(output, dim=1)3
        right_num += (predictions == label).sum().item()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    test_losses.append(loss_value/len(testGenerator))
    accuracy = right_num / len(testDataset)

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    test_f1_scores.append(f1)

    roc_auc = roc_auc_score(label_binarize(all_labels, classes=np.unique(all_labels)),label_binarize(all_predictions, classes=np.unique(all_labels)),average='weighted')
    test_roc_auc.append(roc_auc)
    print(f'ROC AUC: {roc_auc}')

    recall = recall_score(all_labels, all_predictions, average='weighted')
    test_recall.append(recall)
    print(f'Recall: {recall}')


    confusion_mat = confusion_matrix(all_labels, all_predictions)
    print('Confusion Matrix:')
    print(confusion_mat)

    if accuracy > best_right:
        best_right = accuracy
        torch.save(gru_model.state_dict(), './gru_model.pth')
        print('save model')
        early_stop = 0
    else:
        early_stop += 1
        if early_stop > 3:
            print('early stop')
            break

"""DEPLOY MODEL"""
import nest_asyncio
nest_asyncio.apply()

import discord
import torch
import torch.nn as nn
import pandas as pd
from discord.ext import commands

class LabelIndex:
    def __init__(self, dataset):
        self.labels = dataset['type'].unique()
        self.label_index = {label: index for index, label in enumerate(self.labels)}

    def __call__(self, label):
        return self.label_index[label]

class CharIndex:
    def __init__(self, urls):
        self.char_index = {}
        for url in urls:
            for char in url:
                if char not in self.char_index:
                    self.char_index[char] = len(self.char_index)

    def string_indexes(self, string):
        return [self.char_index[char] for char in string]

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(GRUModel, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        if self.bidirectional:
            out = out[:, -1, :self.hidden_size] + out[:, 0, self.hidden_size:]
        return self.fc(out)

class YourAIFunction:
    def __init__(self, gru_model, char_index, label_index):
        self.gru_model = gru_model
        self.char_index = char_index
        self.label_index = label_index

    def preprocess_url(self, url):
        return torch.tensor(self.char_index.string_indexes(url))

    def __call__(self, url):
        if not url:
            return "Empty URL"

        self.gru_model.eval()
        with torch.no_grad():
            inputs = self.preprocess_url(url)
            print("Inputs tensor:", inputs)

            
            if len(inputs) == 0:
                return "Empty URL"

            inputs = inputs.long().unsqueeze(0)
            inputs = inputs.cuda() if next(self.gru_model.parameters()).is_cuda else inputs
            output = self.gru_model(inputs)
            prediction = torch.argmax(output, dim=1).item()
            label = self.label_index.labels[prediction]
        return label

dataset = pd.read_csv('/content/dataset/malicious_phish.csv')
label_index = LabelIndex(dataset)
char_index = CharIndex(dataset['url'])

gru_model = GRUModel(len(char_index.char_index), 128, 128, len(label_index.labels), bidirectional=True, num_layers=1)
gru_model.cuda()

your_ai_function = YourAIFunction(gru_model, char_index, label_index)
TOKEN = "your_discord_bot_token"
CHANNEL = 'your_discord_channel_id'
intents = discord.Intents.all()
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

async def check_url(ctx):
    url = ctx.content
    try:
        result = your_ai_function(url)
        result_message = f'The AI says the URL is {result}.'
        await ctx.send(result_message)
    except Exception as e:
        print(f"An error occurred: {e}")

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

    for guild in bot.guilds:
        for channel in guild.channels:
            print(channel.name, channel.id)

    channel = bot.get_channel(CHANNEL)
    if channel:
        print('Sending message...')
        await channel.send('AI_Hoshino is READY!')
        await channel.send(EMOJI)
        print('Message sent!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    message_content = message.content

    try:
        result = your_ai_function(message_content)
        result_message = f'The AI says the URL is {result}.'
        await message.channel.send(result_message)
    except Exception as e:
        print(f"An error occurred: {e}")

bot.run(TOKEN)
