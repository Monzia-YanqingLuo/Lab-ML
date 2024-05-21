""" ps2_implementation.py

PUT YOUR NAME HERE:
Alexander Schmidt


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

    labels = None
    loss = 0

    for _ in range(max_iter):

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        loss = np.sum(np.min(distances, axis=1))

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
    R = np.zeros((k - 1, X.shape[0]))
    mergeidx = np.zeros((k - 1, 2))
    kmloss = np.zeros(k - 1)
    kmloss[0] = kmeans_crit(X, r)
    for i in reversed(range(k - 1)):
        min_l = None
        min_r = r
        min_x = 0
        min_y = 0
        for x in np.unique(r):
            for y in np.unique(r):
                if x != y:
                    new_r = r
                    new_r[new_r == x] = y
                    new_loss = kmeans_crit(X, new_r)
                    if min_l is None or min_l > new_loss:
                        min_l = new_loss
                        min_r = new_r
                        min_x = x
                        min_y = y
        kmloss[-(i - 1)] = min_l
        mergeidx[-(i - 1)] = [min_x, min_y]
        R[-(i - 1)] = min_r
        r = min_r

    return R, kmloss, mergeidx


def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    pass


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """

    pass


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
        n, d = X.shape
        K = pi.shape[0]
        gamma = np.zeros((n, K))

        # Compute the responsibilities for each data point and each component
        for k in range(K):
            # Compute the probability density function of the Gaussian component
            pdf = multivariate_normal(mean=mu[k], cov=sigma[k]).pdf(X)
            # Compute the weighted probability
            gamma[:, k] = pi[k] * pdf

        # Normalize the responsibilities to get the posterior probabilities
        gamma_sum = gamma.sum(axis=1)[:, np.newaxis]
        gamma = gamma / gamma_sum
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
        n, d = X.shape
        K = gamma.shape[1]

        # Initialize parameters
        mu = np.zeros((n, d))
        sigma = np.zeros((n, d, d))

        # Compute the weighted sum of responsibilities for each component
        Nk = gamma.sum(axis=0)

        # Update the mixture weights
        pi = Nk / n

        # Update the means
        for k in range(K):
            mu[k, :] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / Nk[k]

        # Update the covariance matrices
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / Nk[k]
        return pi, mu, sigma

    n, d = X.shape
    if init_kmeans:
        indices = kmeans(X, k, 100)[1]
    else:
        indices = np.random.choice(n, k, replace=False)
    mu = X[indices]

    # Initialize covariances to identity matrices
    sigma = [np.eye(d) for _ in range(k)]

    # Initialize weights to be equal
    pi = np.full(k, 1 / k)
    log_likelihood = 0
    for l in range(max_iter):
        gamma = expectation(X, pi, mu, sigma)
        prev_log_likelihood = np.sum(np.log(np.dot(gamma, pi.T)))
        pi, mu, sigma = maximization(X, gamma)
        log_likelihood = np.sum(np.log(np.dot(gamma, pi.T)))
        print(f"number of iterations: {l+1}, log likelihood: {log_likelihood}")
        if abs(log_likelihood - prev_log_likelihood) < eps:
            break
    return pi, mu, sigma, log_likelihood


def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass
