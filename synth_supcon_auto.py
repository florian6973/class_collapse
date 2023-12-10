import copy
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

if __name__ == "__main__":
    torch.manual_seed(0) # https://pytorch.org/docs/stable/notes/randomness.html

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    X, y = make_classification(
                n_samples=10000, 
                n_features=2, 
                n_informative=2, 
                n_redundant=0, 
                n_clusters_per_class=2, 
                random_state=42,
                class_sep=2 # 4
            )

    autoencoder_features_nb = 2
    encoder = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, autoencoder_features_nb))
    linear_classifier = nn.Linear(autoencoder_features_nb, 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )



    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_test, y_test)

    # increase batch size for gpu
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(X_test), shuffle=False)

    def display_model(model, data, labels):
        embeddings = model(torch.FloatTensor(data).to(device=device)).detach().cpu().numpy()
        print(embeddings.shape)
        plt.figure(figsize=(10, 10))
        plt.subplot(1,2,1)
        # plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=0.2)
        if autoencoder_features_nb == 2:
            plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, alpha=0.2)
        else:
            plt.scatter(embeddings[:,0], [1]*len(embeddings), c=labels, alpha=0.2)
        plt.subplot(1,2,2)
        plt.scatter(data[:,0], data[:,1], c=labels, alpha=0.2)
        plt.show()

    samples = np.arange(X_test.shape[0])
    X_test_sample = X_test[samples]
    y_test_sample = y_test[samples]
    encoder = encoder.to(device=device)
    display_model(encoder, X_test_sample, y_test_sample)


    class AutoencoderClassifier(L.LightningModule):
        def __init__(self, encoder, linear_classifier):
            super().__init__()
            self.encoder = encoder
            # self.linear_classifier = linear_classifier

        def forward(self, x):
            x = self.encoder(x)
            # x = self.linear_classifier(x)
            return x
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            # print(y_hat.shape)
            # print(y)
            # loss = nn.functional.cross_entropy(y_hat, y)
            # print()
            # print("Pytorch", loss)
            loss = CustomSupConLoss()(y_hat, y)
            # print(y_hat.unsqueeze(1).shape)
            # loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
            # print("Custom", loss2)
            self.log("train_loss", loss)
            return loss
        
        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            # optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
            return optimizer



    # classifier = AutoencoderClassifier(encoder, linear_classifier)
    # trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, strategy="auto")
    # trainer.fit(model=classifier, train_dataloaders=train_dataloader)

    # exit()
    # checkpoint = "./lightning_logs/version_72/checkpoints/epoch=99-step=10500.ckpt"
    from natsort import natsorted
    checkpoint_folder = natsorted(list(os.listdir("./lightning_logs/")))[-1]
    print(checkpoint_folder)
    checkpoint = "./lightning_logs/{}/checkpoints/epoch=8-step=9.ckpt".format(checkpoint_folder)

    # checkpoint = "./lightning_logs/version_62/checkpoints/epoch=99-step=10500.ckpt"
    # 
    # checkpoint = "./lightning_logs/version_54/checkpoints/epoch=99-step=10500.ckpt"
    linear_classifier = AutoencoderClassifier.load_from_checkpoint(checkpoint, encoder=encoder, linear_classifier=linear_classifier)
    # linear_classifier = classifier
    encoder = linear_classifier.encoder
    encoder.eval()
    linear_classifier.eval()

    # samples = np.random.choice(X_test.shape[0], size=int(0.1*X_test.shape[0]), replace=False)

    # plot the embeddings
    # display_model(classifier.encoder, X_test_sample)
    display_model(linear_classifier.encoder, X_test_sample, y_test_sample)

    checkpoint = "./lightning_logs/version_11/checkpoints/epoch=99-step=100.ckpt"
    linear_classifier = AutoencoderClassifier.load_from_checkpoint(checkpoint, encoder=encoder, linear_classifier=linear_classifier)
    # linear_classifier = classifier
    encoder = linear_classifier.encoder
    encoder.eval()
    linear_classifier.eval()
    display_model(linear_classifier.encoder, X_test_sample, y_test_sample)

    # print(autoencoder(em))

    # # predict classes
    # y_hat = linear_classifier(torch.FloatTensor(X_test))
    # y_hat = torch.argmax(y_hat, dim=1)
    # # compute score
    # score = (y_hat == torch.LongTensor(y_test)).sum().item() / len(y_test)
    # print(score)

    # implement supcon loss
    # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/281
    # https://github.com/ivanpanshin/SupCon-Framework
    # pytorch lightning

    # normalize the embeddings? difference loss function?

# https://github.com/RElbers/info-nce-pytorch

# reimplementation manual infonce loss https://github.com/HazyResearch/thanos-code/blob/main/unagi/tasks/loss_fns/contrastive_loss.py

# exact solution interesting as well, directly vectors embeddings to minimize it