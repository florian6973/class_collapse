from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import numpy as np
import os
from torch import optim, nn, utils, Tensor
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mydataset import CustomDataset
import matplotlib.pyplot as plt

torch.manual_seed(0) # https://pytorch.org/docs/stable/notes/randomness.html

autoencoder_features_nb = 1
encoder = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, autoencoder_features_nb))
linear_classifier = nn.Linear(autoencoder_features_nb, 2)

class CustomCELoss(nn.Module):
    def __init__(self):
        super(CustomCELoss, self).__init__()

    def forward(self, predictions, targets):
        assert len(predictions) == len(targets), "Predictions and targets must have the same length."
        log_softmax = nn.LogSoftmax(dim=1)
        log_softmax_predictions = log_softmax(predictions)
        return -torch.sum(log_softmax_predictions[torch.arange(len(predictions)), targets])/len(predictions)

class CustomSupConLoss(nn.Module):
    def __init__(self):
        super(CustomSupConLoss, self).__init__()

    def forward(self, predictions, targets):
        assert len(predictions) == len(targets), "Predictions and targets must have the same length."
        
        # assert len(predictions) == len(targets), "Predictions and targets must have the same length."
        # log_softmax = nn.LogSoftmax(dim=1)
        # log_softmax_predictions = log_softmax(predictions)
        # return -torch.sum(log_softmax_predictions[torch.arange(len(predictions)), targets])/len(predictions)

class AutoencoderClassifier(L.LightningModule):
    def __init__(self, encoder, linear_classifier):
        super().__init__()
        self.encoder = encoder
        self.linear_classifier = linear_classifier

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat.shape)
        # print(y)
        # loss = nn.functional.cross_entropy(y_hat, y)
        # print()
        # print("Pytorch", loss)
        loss = CustomCELoss()(y_hat, y)
        # print("Custom", loss2)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
X, y = make_classification(
            n_samples=10000, 
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

classifier = AutoencoderClassifier(encoder, linear_classifier)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_test, y_test)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# trainer = L.Trainer(max_epochs=100)
# trainer.fit(model=classifier, train_dataloaders=train_dataloader)

# exit()


checkpoint = "./lightning_logs/version_48/checkpoints/epoch=99-step=10500.ckpt"
linear_classifier = AutoencoderClassifier.load_from_checkpoint(checkpoint, encoder=encoder, linear_classifier=linear_classifier)
encoder = linear_classifier.encoder
encoder.eval()
linear_classifier.eval()

# samples = np.random.choice(X_test.shape[0], size=int(0.1*X_test.shape[0]), replace=False)
samples = np.arange(X_test.shape[0])
X_test_sample = X_test[samples]
y_test_sample = y_test[samples]
embeddings = encoder(torch.FloatTensor(X_test_sample)).detach().numpy()
print(embeddings.shape)

# plot the embeddings
plt.figure(figsize=(10, 10))
# plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=0.2)
plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=0.2)
plt.show()
# print(autoencoder(em))

# predict classes
y_hat = linear_classifier(torch.FloatTensor(X_test))
y_hat = torch.argmax(y_hat, dim=1)
# compute score
score = (y_hat == torch.LongTensor(y_test)).sum().item() / len(y_test)
print(score)

# implement supcon loss
# https://github.com/KevinMusgrave/pytorch-metric-learning/issues/281
# https://github.com/ivanpanshin/SupCon-Framework
# pytorch lightning