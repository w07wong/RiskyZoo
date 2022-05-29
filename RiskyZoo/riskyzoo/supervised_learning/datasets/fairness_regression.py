import numpy as np
import matplotlib.pyplot as plt
import torch

"""Generates training and testing data for fairness regression task. Returns tensors of data."""
class FairnessRegression():
    def generate_data(self, n_train, n_test, data_dim, seed=15, display=False, display_path=''):
        np.random.seed(seed)
        
        # Generate train data
        X_train = np.random.normal(loc=0, scale=1, size=(n_train, data_dim))
        eps_train = np.random.normal(loc=0, scale=0.01, size=(n_train, 1))

        theta = np.array([1 for _ in range(data_dim)])
        y_train = np.array([X_train[i].dot(theta) + eps_train[i] if X_train[i][0] <= 1.645 else X_train[i].dot(theta) + X_train[i][0] + eps_train[i] for i in range(n_train)])

        y_train = y_train.reshape(-1, 1)
        X_train = torch.tensor(X_train, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        
        # Generate test data
        X_test = np.random.normal(loc=0, scale=1, size=(n_test, data_dim))
        eps_test = np.random.normal(loc=0, scale=0.01, size=(n_test, 1))

        theta = np.array([1 for _ in range(data_dim)])
        is_majority = [False if X_test[i][0] > 1.645 else True for i in range(n_test)]
        y_test = np.array([X_test[i].dot(theta) + eps_test[i] if X_test[i][0] <= 1.645 else X_test[i].dot(theta) + X_test[i][0] + eps_test[i] for i in range(n_test)])

        if display:
            plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
            plt.savefig(display_path + '/fairness_regression/train.png')
            plt.close()
            plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
            plt.savefig(display_path + '/fairness_regression/test.png')
            plt.close()

        y_test = y_test.reshape(-1, 1)
        
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
        
        return X_train, X_test, y_train, y_test, is_majority