import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from model_utils import gradient_descent
from model_utils import predict


def visualize_data(X_scaled, y):
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X_scaled[:, 1:])  # remove bias term for PCA

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_X[y.ravel() == 0, 0], reduced_X[y.ravel() == 0, 1], color='blue', label='No Disease')
    plt.scatter(reduced_X[y.ravel() == 1, 0], reduced_X[y.ravel() == 1, 1], color='red', label='Heart Disease')
    plt.title("2D Projection of Heart Data (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cost_history(cost_history):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cost_history)), cost_history, label="Cost", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Decrease Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_decision_boundary(X_scaled, y, w, b):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X_scaled[:, 1:])  # drop bias

    w_pca = pca.transform(w[1:].reshape(1, -1))[0]  # project weights
    X_line = np.linspace(X_2D[:, 0].min(), X_2D[:, 0].max(), 100)
    y_line = -(w_pca[0] * X_line + b) / (w_pca[1] + 1e-6)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[y.ravel() == 0, 0], X_2D[y.ravel() == 0, 1], label='No Disease', alpha=0.6)
    plt.scatter(X_2D[y.ravel() == 1, 0], X_2D[y.ravel() == 1, 1], label='Disease', alpha=0.6)
    plt.plot(X_line, y_line, color='black', linestyle='--', label='Decision Boundary')
    plt.title("Decision Boundary in PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
