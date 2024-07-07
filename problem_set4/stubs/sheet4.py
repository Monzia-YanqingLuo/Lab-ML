""" ps4_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from torch.optim import SGD
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
    
    def fit(self, X, Y):

        # INSERT_CODE
        n_samples, n_features = X.shape

        # Compute the Kernel matrix
        K = self.computeKernel(X)
        
        # Here you have to set the matrices as in the general QP problem
        #P = 
        #q = 
        #G = 
        #h = 
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        P = cvxmatrix(np.outer(Y, Y) * K)
        q = cvxmatrix(-np.ones(n_samples))
        
        G_std = np.eye(n_samples) * -1
        G_slack = np.eye(n_samples)
        G = cvxmatrix(np.vstack((G_std, G_slack)))
        
        h_std = np.zeros(n_samples)
        h_slack = np.ones(n_samples) * self.C
        h = cvxmatrix(np.hstack((h_std, h_slack)))
        
        A = cvxmatrix(Y, (1, n_samples), 'd')
        b = cvxmatrix(0.0)
        
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        #b = 
        # Support vectors have non-zero lagrange multipliers
        sv = alpha > 1e-5
        self.alpha_sv = alpha[sv]
        self.X_sv = X[sv]
        self.Y_sv = Y[sv]
        
        # Intercept
        self.b = np.mean([self.Y_sv[i] - np.sum(self.alpha_sv * self.Y_sv * K[ind, sv])
                          for i, ind in enumerate(np.arange(len(alpha))[sv])])

    def predict(self, X):

        if self.alpha_sv is None or self.b is None:
            raise ValueError("Model is not trained yet. Call `fit` first.")
        
        K = self.computeKernel(X, self.X_sv)
        return np.sign(np.dot(K, self.alpha_sv * self.Y_sv) + self.b)
    
    def computeKernel(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        elif self.kernel == 'poly':
            return (1 + np.dot(X, Y.T)) ** self.kernelparameter
        elif self.kernel == 'rbf':
            K = np.zeros((X.shape[0], Y.shape[0]))
            for i, x in enumerate(X):
                K[i, :] = np.exp(-np.linalg.norm(x - Y, axis=1) ** 2 / (2 * (self.kernelparameter ** 2)))
            return K
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1, C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, y, model):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    print("Shape of xx:", xx.shape)
    print("Shape of yy:", yy.shape)
    
    # Prepare to plot the decision boundary
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    print("Total points in grid:", grid_points.shape[0])

    Z = model.predict(grid_points)
    print("Output size of Z before reshape:", Z.size)

    try:
        Z = Z.reshape(xx.shape)
    except ValueError as e:
        print("Error in reshaping Z:", e)
        return

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(('red', 'blue')), label='Data Points')
    if hasattr(model, 'X_sv') and hasattr(model, 'Y_sv'):
        plt.scatter(model.X_sv[:, 0], model.X_sv[:, 1], 
                    s=100, facecolors='none', edgecolors='k', marker='x', 
                    label='Support Vectors')
    plt.legend()
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.show()


def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y.isinstance(bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K

class neural_network(nn.Module):
    def __init__(self, layers=[2, 100, 2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(scale * tr.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = nn.ParameterList([nn.Parameter(scale * tr.randn(n)) for n in layers[1:]])
        # self.weights = None
        # self.biases = None
        self.p = p
        self.lr = lr
        self.lam = lam
        self.train_mode = False

    def relu(self, X, W, b):
        return F.relu(X @ W + b)

    def softmax(self, X, W, b):
        return F.softmax(X @ W + b, dim=1)

    def forward(self, X):
        X = tr.tensor(X, dtype=tr.float)
        for i in range(len(self.weights) - 1):
            X = self.relu(X, self.weights[i], self.biases[i])
            if self.p is not None and self.train_mode:
                X = F.dropout(X, p=self.p)

        X = self.softmax(X, self.weights[-1], self.biases[-1])
        return X

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        ypred = tr.clamp(ypred, 1e-9, 1 - 1e-9)  # To avoid log(0)
        return -tr.mean(tr.sum(ytrue * tr.log(ypred), dim=1))

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.floor(.9 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()
