from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import numpy as np
import os
from torch import optim, nn, utils, Tensor
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from class_collapse.config.config import Config

class SyntheticDataset(Dataset):
    def __init__(self, x, labels):
        self.x = torch.FloatTensor(x)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]
    
def generate_dataset(config: Config) -> (DataLoader, DataLoader):    
    X, y = make_classification(
                n_samples=300, 
                n_features=2, 
                n_informative=2, 
                n_redundant=0, 
                n_clusters_per_class=2, 
                random_state=42,
                class_sep=2 # 4
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    train_dataset = SyntheticDataset(X_train, y_train)
    val_dataset = SyntheticDataset(X_test, y_test)

    batch_size = config.hydra_config["dataset"]["batch_size"]
    if batch_size == -1:
        batch_size = len(X_train)

    # increase batch size for gpu
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(X_test), shuffle=False)

    return train_dataloader, val_dataloader