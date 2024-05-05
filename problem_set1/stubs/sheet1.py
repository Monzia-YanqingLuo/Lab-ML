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
        idx = np.argsort(eigenvalues)[::-1]  # sorted is from the largest to the smallest
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]
        self.D = sorted_eigenvalues
        self.U = sorted_eigenvectors

    def project(self, Xtest, m):
        return np.dot(Xtest - self.C, self.U[:, :m])

    def denoise(self, Xtest, m):
        return np.dot(self.project(Xtest, m), self.U[:, :m].T) + self.C


# assignment 2

from scipy.spatial import distance_matrix
import numpy as np

def gammaidx(X, k):
    n = X.shape[0]  # Number of data points
    D = distance_matrix(X, X)  # Compute the distance matrix
    
    # Initialize the γ-index array
    y = np.zeros(n)
    
    # Iterate over each data point
    for i in range(n):
        # Find the indices of the k-nearest neighbors (excluding the point itself)
        neighbor_ids = np.argsort(D[i])[1:k+1]  # Exclude self in nearest neighbors
        
        # Retrieve the original points for the k-nearest neighbors
        neighbors = X[neighbor_ids]
        
        # Project the neighbors by subtracting the mean of the neighbors
        mean_neighbors = neighbors.mean(axis=0)
        projected_neighbors = neighbors - mean_neighbors
        
        # Compute the mean Euclidean distance from the original point to its projected neighbors
        original_point = X[i] - mean_neighbors  # Also project the original point
        distances = np.linalg.norm(projected_neighbors - original_point, axis=1)
        y[i] = distances.mean()
    
    return y



# assignment 3

def auc(y_true, y_pred, plot=False):
    # Sort predictions and corresponding true values
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    
    # Create list of thresholds (unique y_pred_sorted values)
    thresholds = np.unique(y_pred_sorted)[::-1]
    
    # Initialize lists to store TPR and FPR
    tpr_list = [0]
    fpr_list = [0]
    
    # Compute TPR and FPR for each threshold
    for threshold in thresholds:
        y_pred_thresholded = (y_pred_sorted >= threshold).astype(int) * 2 - 1  # Convert to {-1, 1}
        TP = np.sum((y_pred_thresholded == 1) & (y_true_sorted == 1))
        FP = np.sum((y_pred_thresholded == 1) & (y_true_sorted == -1))
        TN = np.sum((y_pred_thresholded == -1) & (y_true_sorted == -1))
        FN = np.sum((y_pred_thresholded == -1) & (y_true_sorted == 1))
        tpr = TP / (TP + FN) if TP + FN != 0 else 0
        fpr = FP / (FP + TN) if FP + TN != 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Append the end point (1,1) to TPR and FPR lists
    tpr_list.append(1)
    fpr_list.append(1)
    
    # Convert lists to numpy arrays for calculation
    tpr_array = np.array(tpr_list)
    fpr_array = np.array(fpr_list)
    
    # Calculate AUC using trapezoidal rule
    auc_value = np.trapz(tpr_array, fpr_array)
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_array, tpr_array, label="ROC Curve (AUC = {:.2f})".format(auc_value))
        plt.plot([0, 1], [0, 1], 'r--')  # Diagonal reference line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend()
        plt.show()
    
    return auc_value


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
