""" ps2_implementation.py

PUT YOUR NAME HERE:
Alexander Schmidt, Tu Siyuan, Yanqing Luo


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import division  # always use float division
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram


def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (d x k) matrix with each cluster center in one column
    r: assignment vector
    """
    centroids = X[np.random.choice(range(X.shape[0]), k, replace=False)]

    distances = cdist(X, centroids)
    labels = np.argmin(distances, axis=1)
    loss = np.sum(np.min(distances, axis=1))

    for i in range(max_iter):

        old_labels = np.copy(labels)
        new_centroids = np.array(
            [X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else X[np.random.choice(X.shape[0])] for i in
             range(k)])
        distances = cdist(X, new_centroids)
        labels = np.argmin(distances, axis=1)

        loss = np.sum(np.min(distances, axis=1))

        n_changed_clusters = np.array(labels == old_labels)

        print(f"number of iterations: {i}, changed clusters: {len(n_changed_clusters[n_changed_clusters == False])}, "
              f"loss: {loss} ")

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels, loss


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    def kmeans_crit(X, r):
        """ Computes k-means criterion

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector

        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """
        centroids = np.array([X[r == n].mean(axis=0) for n in np.unique(r)])
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        return np.sum(np.min(distances, axis=1))

    k = len(np.unique(r))
    n = k
    R = np.zeros((k - 1, X.shape[0]))
    mergeidx = np.zeros((k - 1, 2))
    kmloss = np.zeros(k)
    kmloss[0] = kmeans_crit(X, r)
    for i in reversed(range(k - 1)):
        min_l = None
        min_r = r
        min_x = 0
        min_y = 0
        for x in np.unique(r):
            for y in np.unique(r):
                if x != y:
                    new_r = np.copy(r)
                    new_r[new_r == x] = n
                    new_r[new_r == y] = n
                    new_loss = kmeans_crit(X, new_r)
                    if min_l is None or min_l > new_loss:
                        min_l = new_loss
                        min_r = np.copy(new_r)
                        min_x = x
                        min_y = y
        kmloss[k - i - 1] = min_l
        mergeidx[k - i - 2] = [min_x, min_y]
        R[k - i - 2] = min_r
        r = min_r
        n += 1

    return R, kmloss, mergeidx


def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    idx = np.array(mergeidx)
    loss = np.array(kmloss[1:])  # Extract loss values
    after_idx = np.max(mergeidx, axis=1) + 1

    # Combine indices, loss values, and newly formed cluster indices
    matrix_with_column = np.column_stack((idx, loss.T, after_idx))

    # Plot the dendrogram
    plt.figure()
    dendrogram(matrix_with_column)
    plt.ylim(kmloss[0], )
    plt.title('Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.show()


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """
    d = X.shape[1]

    # Compute the determinant and the inverse of the covariance matrix
    C += 1e-6 * np.eye(C.shape[0])
    try:
        inv_C = np.linalg.inv(C)
        det_C = np.linalg.det(C)
    except np.linalg.LinAlgError:
        # If the matrix is still singular, return a PDF value of zero
        return 1e-6

    if det_C == 0:
        return 1e-6
    # compute coefficient
    coeff = 1 / (np.sqrt((2 * np.pi) ** d * det_C))

    # compute X_i - mu
    diff = X - mu

    # Compute the exponent term for each data point
    exponent = -0.5 * np.sum(diff @ inv_C * diff, axis=1)

    # Compute the PDF values
    pdf_values = coeff * np.exp(exponent)

    return pdf_values


def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    def expectation(X, pi, mu, sigma):
        """
        Expectation step of the EM algorithm.

        Args:
          X: The data points.
          pi: The weights of each Gaussian component.
          mu: The means of the Gaussian components.
          sigma: The covariance matrices of the Gaussian components.

        Returns:
          A matrix containing the responsibilities (posterior probabilities)
          of each data point belonging to each Gaussian component.
        """
        N, D = X.shape
        K = len(pi)

        # Initialize responsibility matrix
        gamma = np.zeros((N, K))

        # Compute the responsibility for each data point and component
        for k in range(K):
            rv = norm_pdf(X, mu[k], sigma[k])
            gamma[:, k] = pi[k] * rv

        # Normalize responsibilities to sum to 1 for each data point
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        return gamma

    def maximization(X, gamma):
        """
        Maximization step of the EM algorithm.

        Args:
          X: The data points.
          gamma: The responsibilities (posterior probabilities)
               of each data point belonging to each Gaussian component.

        Returns:
          Updated weights, means, and covariance matrices for the Gaussian components.
        """
        N, D = X.shape
        K = gamma.shape[1]

        # Compute the effective number of points assigned to each component
        Nk = gamma.sum(axis=0)

        # Update the mixture weights
        pi = Nk / N

        # Update the means
        mu = (gamma.T @ X) / Nk[:, np.newaxis]

        # Update the covariance matrices
        sigma = np.zeros((K, D, D))
        for k in range(K):
            diff = X - mu[k]
            weighted_diff = gamma[:, k][:, np.newaxis] * diff
            sigma[k] = weighted_diff.T @ diff / Nk[k]

        return pi, mu, sigma

    def log_likelihood_f(X, pi, mu, sigma):

        N, D = X.shape
        K = len(pi)

        # Initialize log-likelihood array
        ll = np.zeros(N)

        # Compute the log likelihood for each component and each data point
        for k in range(K):
            rv = norm_pdf(X, mu[k], sigma[k])
            ll += pi[k] * rv

        # Compute the total log likelihood
        total_log_likelihood = np.sum(np.log(ll))

        return total_log_likelihood

    n, d = X.shape
    if init_kmeans:
        mu, labels, _ = kmeans(X, k, 100)
        sigma = []
        for i in range(k):
            cluster_data = X[labels == i]
            if len(cluster_data) > 1:
                cluster_covariance = np.cov(cluster_data, rowvar=False)
            else:
                cluster_covariance = np.identity(X.shape[1])  # Handle clusters with a single data point
            sigma.append(cluster_covariance)
    else:
        indices = np.random.choice(n, k, replace=False)
        mu = X[indices]
        data_covariance = np.cov(X, rowvar=False)

        # Initialize each component's covariance matrix with the overall data covariance matrix
        sigma = np.array([data_covariance for _ in range(k)])

    # Initialize weights to be equal
    pi = np.full(k, 1 / k)
    log_likelihood = 0
    for l in range(max_iter):
        gamma = expectation(X, pi, mu, sigma)
        prev_log_likelihood = log_likelihood_f(X, pi, mu, sigma)
        pi, mu, sigma = maximization(X, gamma)
        log_likelihood = log_likelihood_f(X, pi, mu, sigma)
        print(f"number of iterations: {l + 1}, log likelihood: {log_likelihood}")
        if abs(log_likelihood - prev_log_likelihood) < eps:
            break
    return pi, mu, sigma, log_likelihood


def plot_gmm_solution(X, mu, sigma, n_std=1.0):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    plt.scatter(X[0], X[1])
    for k in range(mu.shape[0]):
        # Plot the means
        plt.scatter(mu[k, 0], mu[k, 1], marker='x', s=100, label=f'Mean {k + 1}')

        # Plot the covariance as ellipses
        eigenvalues, eigenvectors = np.linalg.eigh(sigma[k])
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        ell = Ellipse(xy=mu[k], width=width, height=height, angle=angle, color="red", alpha=0.5)
        plt.gca().add_patch(ell)
