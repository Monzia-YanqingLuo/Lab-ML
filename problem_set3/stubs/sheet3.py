""" sheet3.py

PUT YOUR NAME HERE:
Alexander Schmidt, Tu Siyuan, Yanqing Luo


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import time
import itertools
import numpy as np


def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''
    if not isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        print(f"Both inputs must be np-array.")
    return np.sum(y_pred != y_true) / len(y_true)


def mean_absolute_error(y_true, y_pred):
    ''' your code here '''
    if not isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        print(f"Both inputs must be np-array.")

    return sum(abs(y_true - y_pred)) / len(y_pred)


class KFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)

        rng = np.random.default_rng(True)
        rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        splits = []

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate((indices[:start], indices[stop:]))
            splits.append((train_idx, test_idx))
            current = stop

        return splits


def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    best_loss = float('inf')
    best_params = None
    best_model = None
    kf = KFold(n_splits=nfolds)

    # all_param_combinations = list(itertools.product(*parameters.values()))

    # all_param_combinations = np.array(list(itertools.product(params['regularization'], params['kernel'])))
    param_keys = list(params.keys())
    all_param_combinations = list(itertools.product(*(params[key] for key in param_keys)))

    total_iterations = len(all_param_combinations) * nrepetitions * nfolds
    iteration = 0
    start_time = time.time()

    for param_combination in all_param_combinations:
        # param_dict = dict(zip(['regularization', 'kernel'], param_combination))
        # param_dict['kernelparameter'] = params['kernelparameter']
        param_dict = dict(zip(param_keys, param_combination))

        total_loss = 0

        for _ in range(nrepetitions):
            fold_losses = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model = method(**param_dict)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                fold_loss = loss_function(y_val, y_pred)
                fold_losses.append(fold_loss)

                # Update iteration and report progress
                iteration += 1
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / iteration) * (total_iterations - iteration)
                print(f'Progress: {iteration}/{total_iterations} - '
                      f'Elapsed Time: {elapsed_time:.2f}s - '
                      f'Remaining Time: {remaining_time:.2f}s', end='\r')

            total_loss += np.mean(fold_losses)

        avg_loss = total_loss / nrepetitions

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = param_dict
            best_model = method(**best_params)
            best_model.fit(X, y)

    best_model.cvloss = best_loss
    return best_model


class krr():
    ''' Kernel Ridge Regression (KRR) implements ridge regression with various kernel functions.
        This class allows specification of the kernel type, kernel parameters, and regularization strength.
    '''

    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.alpha = None  # Coefficient matrix for predictions
        self.X_train = None  # Store training features here

    def _compute_kernel(self, X1, X2):
        ''' Compute the kernel matrix between datasets X1 and X2 based on the specified kernel type. '''
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1).reshape(1, -1) - 2 * np.dot(X1,
                                                                                                                    X2.T)
            e = -0.5 / self.kernelparameter ** 2 * sq_dists
            return np.exp(e)
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' Fit the KRR model using the provided training data and labels.
            X: Feature matrix.
            y: Target vector.
            Optionally, kernel type, kernel parameter, and regularization can be adjusted.
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        self.X_train = X
        n = X.shape[0]
        K = self._compute_kernel(X, X)
        I = np.eye(n)
        self.alpha = np.linalg.inv(K + self.regularization * I) @ y

        return self

    def predict(self, X):
        ''' Predicts the target values for given input features using the trained KRR model.
            X: New feature matrix for which predictions are to be made.
            Raises ValueError if the model has not been fitted.
        '''
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' with appropriate data.")

        K_new = self._compute_kernel(X, self.X_train)
        return np.dot(K_new, self.alpha)
