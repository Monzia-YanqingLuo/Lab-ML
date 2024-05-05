import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt


# assignment 1
class PCA:
    def __init__(self, Xtrain):
        self.Xtrain = Xtrain
        m = Xtrain.shape[1]
        self.C = np.mean(Xtrain, axis=0)
        X_centered = Xtrain - self.C
        Cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(Cov)
        idx = np.argsort(eigenvalues)[::-1]  # sorted is from the smallest to the largest
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]
        self.D = sorted_eigenvalues
        self.U = sorted_eigenvectors

    def project(self, Xtest, m):
        return np.dot(Xtest - self.C, self.U[:, :m])

    def denoise(self, Xtest, m):
        return np.dot(self.project(Xtest, m), self.U[:, :m].T) + self.C


# assignment 2

def gammaidx(X, k):
    n = X.shape[0]  # Number of data points as shown in Table1
    D = distance_matrix(X, X)  # Compute the distance matrix

    # Initialize the γ-index array
    y = np.zeros(n)

    # Iterate over each data point
    for i in range(n):
        # Find the indices of the k-nearest neighbors (including the point itself)
        neighbor_ids = np.argsort(D[i])[:k + 1]  # Get the k+1 smallest distances (including itself)

        # Retrieve the original points for the k-nearest neighbors
        neighbors = X[neighbor_ids]

        # Project the neighbors: This step involves PCA or any dimensionality reduction method
        # Here we are assuming a simple mean subtraction is the 'projection'
        projected_neighbors = neighbors - neighbors.mean(axis=0)

        # Compute the mean Euclidean distance from the point to its projected neighbors
        y[i] = np.mean(np.linalg.norm(X[i] - projected_neighbors, axis=1))

    return y


# assignment 3

def auc(y_true, y_pred, plot=False):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == -1))
    TN = np.sum((y_pred == -1) & (y_true == -1))
    FN = np.sum((y_pred == -1) & (y_true == 1))
    tpr = TP / (TP + FN) if TP + FP != 0 else 0
    fpr = FP / (FP + TN) if FP + TN != 0 else 0
    auc = np.trapz(tpr, fpr)
    if plot == True:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("False positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend()
        plt.show()
    else:
        return auc


# assignment 4
def lle(X, m, n_rule, k=None, epsilon=None, tol=1e-2):
    n = X.shape[0]
    D = distance_matrix(X, X)
    if n_rule == 'knn':
        neighbors = np.argsort(D, axis=1)[:, 1:k + 1]
    elif n_rule == 'eps-ball':
        neighbors = [np.where(D[i] <= epsilon)[0] for i in range(n)]
    else:
        raise ValueError("Invalid neighborhood rule specified.")
    if k == 0 or not all(len(x) > 1 for x in neighbors):
        raise ValueError("Graph is not connected")
    w_initial = np.zeros((n, n))
    for i in range(n):
        k_i = neighbors[i]
        Z = X[k_i] - X[i]
        C = np.dot(Z, Z.T)
        w = np.linalg.solve(C + tol * np.eye(len(k_i)), np.ones(len(k_i)))
        w_initial[i, k_i] = w / np.sum(w)

    M = np.eye(n) - w_initial
    M = np.dot(M.T, M)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = np.argsort(eigenvalues)[1:m + 1]
    return eigenvectors[:, idx]
