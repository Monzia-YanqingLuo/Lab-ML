import numpy as np
from scipy.spatial import distance_matrix


#assignment 4
def lle(X, m, n_rule, k=None, epsilon = None, tol=1e-2):
    n = X.shape[0]
    D = distance_matrix(X, X)
    if n_rule == 'knn':
        neighbors = np.argsort(D, axis=1)[:, 1:k + 1]
    elif n_rule == 'eps-ball':
        neighbors = [np.where(D[i] <= epsilon)[0] for i in range(n)]
    else:
        raise ValueError("Invalid neighborhood rule specified.")
    if k == 0 or not all(len(x)>1 for x in neighbors):
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
