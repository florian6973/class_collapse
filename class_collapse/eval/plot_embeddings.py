from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from class_collapse.config.config import Config
import numpy as np

def plot_embeddings(config: Config, model, test_dataloader, comment=""):
    losses = model.loss_values
    print(losses)
    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(config.hydra_config["loss"]["name"] + " " + comment)
    plt.savefig(f"loss_{comment}.png")

    data_loader = next(iter(test_dataloader))
    data = data_loader[0].detach().cpu().numpy()
    labels = data_loader[1].detach().cpu().numpy()
    print(data_loader)
    embeddings = model(data_loader[0]).detach().cpu().numpy()
    # data = data_loader.x.detach().cpu().numpy()
    # labels = data_loader.labels.detach().cpu().numpy()
    print(embeddings.shape)
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    # plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=0.2)
    if config.hydra_config["model"]["embeddings_features"] == 2:
        plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, alpha=0.2)
    else:
        plt.scatter(embeddings[:,0], [1]*len(embeddings), c=labels, alpha=0.2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(config.hydra_config["model"]["name"] + " " + comment)
    plt.subplot(1,2,2)
    plt.scatter(data[:,0], data[:,1], c=labels, alpha=0.2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Groundtruth")
    plt.savefig(f"embeddings_{comment}.png")