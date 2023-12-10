from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import numpy as np
import os
from torch import optim, nn, utils, Tensor
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from class_collapse.config.config import Config
    
def get_synthetic_dataset(config: Config):  
    X, y = make_classification(
                n_samples=300, 
                n_features=2, 
                n_informative=2, 
                n_redundant=0, 
                n_clusters_per_class=2, 
                random_state=42,
                class_sep=2 # 4
            )
    
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm.fit(X)

    cluster_assignments = gmm.predict(X)
    
    return X, y, cluster_assignments