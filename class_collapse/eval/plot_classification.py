from class_collapse.training.classifier import KNNClassifier
from matplotlib import pyplot as plt
import numpy as np

from class_collapse.eval.plot_embeddings import compute_umap

def plot_classification(config, model, data, comment):
    if data.X_test.shape[1] > 2:
        data_emb = compute_umap(data.X_test)
    else:
        data_emb = data.X_test

    knn = KNNClassifier(model, data)
    _, data.y_test_fine = knn.fit()
    
    plt.figure(figsize=(20, 10))

    plt.subplot(2,2,1)
    plt.scatter(data_emb[:,0], data_emb[:,1], c=data.y_test_fine, cmap='flag')
    plt.title("Fine clusters", fontsize=20)

    plt.subplot(2,2,2)
    plt.scatter(data_emb[:,0], data_emb[:,1], c=data.y_test, cmap='flag')
    plt.title("Coarse clusters", fontsize=20)

    data.X_test_embs, y_pred_fine, y_pred_coarse = knn.predict()
    plt.subplot(2,2,3)
    plt.scatter(data.X_test_embs[:,0], data.X_test_embs[:,1], c=y_pred_fine, cmap='flag')
    plt.title(f"KNN accuracy: {np.mean(y_pred_fine == knn.data.y_test_fine):.3f} ({comment})", fontsize=20)


    plt.subplot(2,2,4)
    plt.scatter(data.X_test_embs[:,0], data.X_test_embs[:,1], c=y_pred_coarse, cmap='flag')    
    plt.title(f"KNN accuracy: {np.mean(y_pred_coarse == knn.data.y_test):.3f} ({comment})", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"clusters_{comment}.png")
