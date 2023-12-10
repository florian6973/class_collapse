from class_collapse.training.classifier import KNNClassifier
from matplotlib import pyplot as plt
import numpy as np

def plot_classification(config, model, X_train, y_train, X_test, y_test, comment):
    knn = KNNClassifier(model, X_train, X_test, y_train, y_test)
    _, y_test_fine = knn.fit()
    
    plt.figure(figsize=(20, 10))

    plt.subplot(2,2,1)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test_fine)
    plt.title("Fine clusters")

    plt.subplot(2,2,2)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
    plt.title("Coarse clusters")

    X_test_embs, y_pred_fine, y_pred_coarse = knn.predict()
    plt.subplot(2,2,3)
    plt.scatter(X_test_embs[:,0], X_test_embs[:,1], c=y_pred_fine)
    plt.title(f"KNN accuracy: {np.mean(y_pred_fine == knn.y_test_fine):.3f} ({comment})")


    plt.subplot(2,2,4)
    plt.scatter(X_test_embs[:,0], X_test_embs[:,1], c=y_pred_coarse)    
    plt.title(f"KNN accuracy: {np.mean(y_pred_coarse == knn.y_test):.3f} ({comment})")

    plt.tight_layout()
    plt.savefig(f"clusters_{comment}.png")
