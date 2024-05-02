import numpy as np
from scipy.spatial import distance_matrix


def lle(X, m, n_rule, param, tol=1e-2):
    n = X.shape[0]
    D = distance_matrix(X, X)
    if n_rule == 'knn':
        neighbors = np.argsort(D, axis=1)[:, 1:param + 1]
    elif n_rule == 'eps-ball':
        neighbors = [np.where(D[i] <= param)[0] for i in range(n)]
    else:
        raise ValueError("Invalid neighborhood rule specified.")

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
    idx = np.argsort(eigenvalues)[1:m + 1]  # ignore the first smallest zero eigenvalue
    return eigenvectors[:, idx]
