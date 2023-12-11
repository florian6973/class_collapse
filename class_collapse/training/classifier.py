from torch import nn, optim
import lightning as L
from class_collapse.config.config import Config
from class_collapse.training.losses import CustomInfoNCELoss, SupConLoss, CustomCELoss
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import torch

class KNNClassifier:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.knn_fine = KNeighborsClassifier(n_neighbors=3)
        self.knn_coarse = KNeighborsClassifier(n_neighbors=3)

    def get_embeddings(self, x):
        embeddings = self.model(torch.Tensor(x)).detach().cpu().numpy()
        return embeddings

    def fit(self):        
        self.y_train_fine = self.data.y_train_fine
        self.y_test_fine = self.data.y_test_fine

        self.knn_fine.fit(self.get_embeddings(self.data.X_train), self.data.y_train_fine)
        self.knn_coarse.fit(self.get_embeddings(self.data.X_train), self.data.y_train)

        return self.data.X_test, self.y_test_fine

    def predict(self):
        X_test_embs = self.get_embeddings(self.data.X_test)
        y_pred_fine = self.knn_fine.predict(X_test_embs)
        y_pred_coarse = self.knn_coarse.predict(X_test_embs)
        return X_test_embs, y_pred_fine, y_pred_coarse
