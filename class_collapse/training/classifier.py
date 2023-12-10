from torch import nn, optim
import lightning as L
from class_collapse.config.config import Config
from class_collapse.training.losses import CustomInfoNCELoss, CustomSupConLoss, SupConLoss, CustomCELoss

from torch.utils.data import Dataset, DataLoader

class Classifier(L.LightningModule):
    def __init__(self, encoder, linear_classifier, config: Config):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.linear_classifier = linear_classifier
        self.loss_values = []
        self.current_loss_values = []
        self.config = config
        # self.linear_classifier = linear_classifier

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # y_hat = self(x)
        # if self.config.hydra_config["loss"]["name"] == "MSE_loss":
        #     x_hat = self.decoder(y_hat)
        #     loss = nn.functional.mse_loss(x_hat, x)
        # # elif self.config.hydra_config["loss"]["name"] == "CE_loss":
        # #     loss = nn.functional.cross_entropy(y_hat, y)
        # elif self.config.hydra_config["loss"]["name"] == "supcon_2020":
        #     loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
        # elif self.config.hydra_config["loss"]["name"] == "nce":
        #     loss = CustomInfoNCELoss()(y_hat, y) # diverges
        # elif self.config.hydra_config["loss"]["name"] == "spread":
        #     alpha = self.config.hydra_config["loss"]["alpha"]
        #     loss = (1-alpha)*SupConLoss()(y_hat.unsqueeze(1), labels=y) + \
        #             alpha*CustomInfoNCELoss()(y_hat, y)
        # else:
        #     raise ValueError("Unknown loss")
        # print(y_hat.shape)
        # print(y)
        # loss = nn.functional.cross_entropy(y_hat, y)
        # print()
        # print("Pytorch", loss)
        # loss = CustomSupConLoss()(y_hat, y)
        # print(y_hat.unsqueeze(1).shape)
        # loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
        # print("Custom", loss2)
        self.log("train_loss", loss)
        self.current_loss_values.append(loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.hydra_config["model"]["lr"])
        # optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer
    
    def on_train_epoch_end(self):
        for loss in self.current_loss_values:
            self.loss_values.append(loss.item())
        self.current_loss_values.clear()  # free memory'

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
    
    # vs y_test_fine