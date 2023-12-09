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


if __name__ == "__main__":
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=10000, 
            n_features=2, 
            n_informative=2, 
            n_redundant=0, 
            n_clusters_per_class=2, 
            random_state=42,
            class_sep=2 # 4
        )
        print(y)

        # plot in 2d, color by class
        import matplotlib.pyplot as plt
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.show()

        # Use KMeans for clustering
        # kmeans = KMeans(n_clusters=4, random_state=42)
        # kmeans.fit(X)
        # Get cluster assignments
        # cluster_assignments = kmeans.labels_
        # use gaussian mixture model
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm.fit(X)
        cluster_assignments = gmm.predict(X)


        # Now, cluster_assignments contains the cluster assignments for each sample in X
        print(cluster_assignments)

        # Plot the clusters
        plt.scatter(X[:,0], X[:,1], c=cluster_assignments)
        # plt.show()

        # train a classifier on the clusters
        # split the data into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, cluster_assignments, test_size=0.33, random_state=42
        )
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        print(lr.score(X_train, y_train))
        print(lr.score(X_test, y_test))

        # train a classifier on the original labels
        # split the data into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        print(lr.score(X_train, y_train))
        print(lr.score(X_test, y_test))

        # train a simple MLP on the clusters
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(1,1))
        # print mlp structure
        print(mlp)
        mlp.fit(X_train, y_train)
        print(mlp.score(X_train, y_train))
        print(mlp.score(X_test, y_test))
        print(mlp.coefs_)
        print(mlp.intercepts_)
        print(mlp.n_layers_)
        print(mlp.n_features_in_)
        print(mlp.n_outputs_)


        # supcon python package
        # import matplotlib.pyplot as plt

        # from mlxtend.plotting import plot_nn_graph
        # from sklearn.neural_network import MLPClassifier

        # # Visualize the MLP architecture
        # fig = plt.figure(figsize=(12, 8))
        # plot_nn_graph(mlp, filename=None, show=True)
        # plt.show()

        import graphviz

        # Visualize the neural network architecture using graphviz
        dot_data = f"digraph MLP {{rankdir=LR;"

        # Input layer
        dot_data += "Input [shape=plaintext]\n"
        for i in range(X_train.shape[1]):
            dot_data += f"Input -> X{i} [label=\"X{i}\"]\n"

        # Hidden layers
        for i, coef in enumerate(mlp.coefs_):
            for j in range(coef.shape[0]):
                for k in range(coef.shape[1]):
                    dot_data += f"X{j} -> H{i}_{k} [label=\"W{i}_{k}{j}\"]\n"

        # Output layer
        dot_data += "Output [shape=plaintext]\n"
        for i in range(mlp.coefs_[-1].shape[0]):
            dot_data += f"H{len(mlp.coefs_)-1}_{i} -> Output [label=\"Wout_{i}\"]\n"

        dot_data += "}"

        # Render the graph
        # graph = graphviz.Source(dot_data)
        # graph.render("mlp_graph", format="png", cleanup=True)
        # graph.view("mlp_graph")



        # define any number of nn.Modules (or use your current ones)
        autoencoder_features_nb = 2
        encoder = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, autoencoder_features_nb))
        decoder = nn.Sequential(nn.Linear(autoencoder_features_nb, 5), nn.ReLU(), nn.Linear(5, 2))

        # define the loss function
        # define the LightningModule
        class LitAutoEncoder(L.LightningModule):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def training_step(self, batch, batch_idx):
                # training_step defines the train loop.
                # it is independent of forward
                x, y = batch
                x = x.view(x.size(0), -1)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = nn.functional.mse_loss(x_hat, x)
                # Logging to TensorBoard (if installed) by default
                self.log("train_loss", loss)
                return loss

            def configure_optimizers(self):
                optimizer = optim.Adam(self.parameters(), lr=1e-3)
                return optimizer
            
       

        # init the autoencoder
        autoencoder = LitAutoEncoder(encoder, decoder)

        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_test, y_test)


        # Create PyTorch Lightning DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        trainer = L.Trainer(max_epochs=100)
        trainer.fit(model=autoencoder, train_dataloaders=train_dataloader)

        # encoder = autoencoder.encoder
        # encoder.eval()

        # get the latent space
        # z = encoder(torch.FloatTensor(X))
        # print(z.shape)

        # checkpoint = "./lightning_logs/version_3/checkpoints/epoch=99-step=10500.ckpt"
        # autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

        # choose your trained nn.Module
        encoder = autoencoder.encoder
        encoder.eval()
        decoder = autoencoder.decoder
        decoder.eval()

        # embed 4 fake images!
        # fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
        # embeddings = encoder(fake_image_batch)

        # compute embedding for 10% samples of the test dataset
        samples = np.random.choice(X_test.shape[0], size=int(0.1*X_test.shape[0]), replace=False)
        X_test_sample = X_test[samples]
        y_test_sample = y_test[samples]
        embeddings = encoder(torch.FloatTensor(X_test_sample)).detach().numpy()
        print(embeddings.shape)

        # plot the embeddings
        plt.figure(figsize=(10, 10))
        # plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=0.2)
        plt.scatter(embeddings[:,0], embeddings[:,1], c=y_test_sample, alpha=0.2)
        plt.show()

        decodeds = decoder(torch.FloatTensor(embeddings)).detach().numpy()
        print(decodeds.shape)

        # plot the decodeds
        plt.figure(figsize=(10, 10))
        plt.subplot(1,2,1)
        plt.scatter(decodeds[:,0], decodeds[:,1], c=y_test_sample, alpha=0.2)
        plt.subplot(1,2,2)
        plt.scatter(X_test_sample[:,0], X_test_sample[:,1], c=y_test_sample, alpha=0.2)
        plt.show()
