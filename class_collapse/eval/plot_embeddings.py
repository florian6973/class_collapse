from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from class_collapse.config.config import Config
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
import umap

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
            # print(label)
            # print(embeddings[labels==label].shape)
            cs_e = F.cosine_similarity(torch.Tensor(embeddings[labels==label])[None, :, :], 
                                    torch.Tensor(embeddings[labels==label])[:, None, :], dim=2)
            # print(cs_e.shape)
            cs.append(cs_e.mean().item())
        return cs        

    alpha_val = 0.8
    losses = model.loss_values
    # print(losses)
    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    # plot in log scale
    # plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(config.hydra_config["loss"]["name"] + " " + comment)
    plt.savefig(f"loss_{comment}.png")

    labels = data_loader[1].detach().cpu().numpy()

    # cos_sine = nn.CosineSimilarity(dim=1, eps=1e-6))

    # print(data_loader)
    embeddings_t = model(data_loader[0])
    embeddings = embeddings_t.detach().cpu().numpy()
    # embeddings[labels == 0] = 0 + np.random.normal(1, 0.1, size=(len(embeddings[labels == 0]), config.hydra_config["model"]["embeddings_features"]))
    # cs_e = cos_sine(embeddings, embeddings)
    
    # cs_e = F.cosine_similarity(embeddings_t[None, :, :], embeddings_t[:, None, :], dim=2)
    # print(cs_e.shape)
    # exit()
    cs_e = cosine_similarities(embeddings, labels)
    cs_d = cosine_similarities(data, labels)
    # data = data_loader.x.detach().cpu().numpy()
    # labels = data_loader.labels.detach().cpu().numpy()
    # print(embeddings.shape)
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    # plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=alpha_val)
    if config.hydra_config["model"]["embeddings_features"] == 2:
        plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, alpha=alpha_val)
    else:
        plt.scatter(embeddings[:,0], [1]*len(embeddings), c=labels, alpha=alpha_val)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(config.hydra_config["model"]["name"] + " " + comment + " " + ",".join([str(round(x, 2)) for x in cs_e]))
    plt.subplot(1,2,2)
    plt.scatter(data_emb[:,0], data_emb[:,1], c=labels, alpha=alpha_val)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Groundtruth " + ",".join([str(round(x, 2)) for x in cs_d]))
    plt.savefig(f"embeddings_{comment}.png")