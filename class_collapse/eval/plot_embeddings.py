from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from class_collapse.config.config import Config
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
import umap

import matplotlib
matplotlib.rcParams.update({'font.size': 20})

def compute_umap(x):
    reducer = umap.UMAP()
    return reducer.fit_transform(x)

def plot_embeddings(config: Config, model, test_dataloader, comment=""):
    data_loader = next(iter(test_dataloader))
    data = data_loader[0].detach().cpu().numpy()

    if data.shape[1] > 2:
        data_emb = compute_umap(data)
    else:
        data_emb = data

    # compute average intraclass cosine similarity
    def cosine_similarities(embeddings, labels):
        cs = []
        for label in np.unique(labels):
            cs_e = F.cosine_similarity(torch.Tensor(embeddings[labels==label])[None, :, :], 
                                    torch.Tensor(embeddings[labels==label])[:, None, :], dim=2)
            cs.append(cs_e.mean().item())
        return cs        

    alpha_val = 0.8
    losses = model.loss_values
    plt.figure(figsize=(10, 10))
    plt.plot(losses)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(config.hydra_config["loss"]["name"] + " " + comment)
    plt.savefig(f"loss_{comment}.png")

    labels = data_loader[1].detach().cpu().numpy()

    embeddings_t = model(data_loader[0])
    embeddings = embeddings_t.detach().cpu().numpy()

    cs_e = cosine_similarities(embeddings, labels)
    cs_d = cosine_similarities(data, labels)

    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    
    if config.hydra_config["model"]["embeddings_features"] == 2:
        plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, alpha=alpha_val, cmap='flag')
    else:
        plt.scatter(embeddings[:,0], [1]*len(embeddings), c=labels, alpha=alpha_val, cmap='flag')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(config.hydra_config["model"]["name"] + " " + comment + " " + ",".join([str(round(x, 2)) for x in cs_e]), fontsize=20)
    plt.subplot(1,2,2)
    plt.scatter(data_emb[:,0], data_emb[:,1], c=labels, alpha=alpha_val, cmap='flag')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Groundtruth " + ",".join([str(round(x, 2)) for x in cs_d]), fontsize=20)
    plt.savefig(f"embeddings_{comment}.png")

    
    plt.figure(figsize=(9, 9))
    plt.scatter(data_emb[:,0], data_emb[:,1], c=labels, alpha=alpha_val, cmap='flag')
    # plt.title("Groundtruth " + ",".join([str(round(x, 2)) for x in cs_d]), fontsize=20)
    cs_title = f"Cosine similarity cluster 1: {round(cs_d[0], 2)}\nCosine similarity cluster 2: {round(cs_d[1], 2)}"
    assert len(cs_d) == 2
    plt.title("Groundtruth\n" + cs_title, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"embeddings_{comment}_groundtruth.png")

    
    plt.figure(figsize=(9, 9))
    plt.scatter(embeddings[:,0], embeddings[:,1] if config.hydra_config["model"]["embeddings_features"] == 2 else [1]*len(embeddings), c=labels, alpha=alpha_val, cmap='flag')
    # plt.title("Groundtruth " + ",".join([str(round(x, 2)) for x in cs_d]), fontsize=20)
    cs_title = f"Cosine similarity cluster 1: {round(cs_e[0], 2)}\nCosine similarity cluster 2: {round(cs_e[1], 2)}"
    assert len(cs_d) == 2
    plt.title(comment.capitalize() + "\n" + cs_title, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"embeddings_{comment}_learnt.png")