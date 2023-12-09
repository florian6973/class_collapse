from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import numpy as np
import os
from torch import optim, nn, utils, Tensor
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, x, labels):
        self.x = torch.FloatTensor(x)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]