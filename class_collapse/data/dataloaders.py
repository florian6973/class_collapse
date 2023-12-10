
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
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
from class_collapse.data.synthetic_dataset import get_synthetic_dataset
from class_collapse.data.house_dataset import get_house_dataset

class LabeledDataset(Dataset):
    def __init__(self, x, labels):
        self.x = torch.FloatTensor(x)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]
    
@dataclass
class DataWrapper:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_train_fine: np.ndarray
    y_test: np.ndarray
    y_test_fine: np.ndarray
    train_dataloader: DataLoader
    test_dataloader: DataLoader

def make_dataloader(config):
    if config.hydra_config["dataset"]["name"] == "synthetic":
        X, y, y_fine = get_synthetic_dataset(config)
    elif config.hydra_config["dataset"]["name"] == "house":
        X, y, y_fine = get_house_dataset(config)

    X_train, X_test, y_train_all, y_test_all = train_test_split(
        X, np.stack((y, y_fine), axis=-1), test_size=0.33, random_state=42
    )

    y_train = y_train_all[:, 0]
    y_train_fine = y_train_all[:, 1]
    y_test = y_test_all[:, 0]
    y_test_fine = y_test_all[:, 1]

    train_dataset = LabeledDataset(X_train, y_train)
    val_dataset = LabeledDataset(X_test, y_test)

    batch_size = config.hydra_config["dataset"]["batch_size"]
    if batch_size == -1:
        batch_size = len(X_train)

    # increase batch size for gpu
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(X_test), shuffle=False)

    return DataWrapper(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_train_fine=y_train_fine,
        y_test=y_test,
        y_test_fine=y_test_fine,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
    )