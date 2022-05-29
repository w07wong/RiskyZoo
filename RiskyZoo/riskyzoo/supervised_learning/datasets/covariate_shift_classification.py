import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

"""Generates training and testing data for covariate shift classification task. Returns tensors of data."""
class CovariateShiftClassification():
    """Method to call to generate data."""
    def generate_data(self, n_samples_1, n_samples_2, noise_level, seed=15, display=False, display_path=''):
        n_samples_1, n_samples_3 = int(n_samples_1 / 2), int(n_samples_1 / 2)
        n_samples_2, n_samples_4 = int(n_samples_2 / 2), int(n_samples_2 / 2)
        centers = [[0.0, 0.0], [0.0, 1.5], [1.5, 1.5], [1.5, 0.0]]
        clusters_std = [0.4, 0.4, 0.4, 0.4]
        X, y = make_blobs(n_samples=[n_samples_1, n_samples_2, n_samples_3, n_samples_4],
                        centers=centers,
                        cluster_std=clusters_std,
                        random_state = 15,
                        shuffle=False)
        
        # Add x1 * x2 to data
        nonlinearity = X[:, 0] * X[:, 1]
        nonlinearity = nonlinearity[:, None]
        X = np.hstack((X, nonlinearity))
        
        # Set diagnoals to be the same class
        y = y % 2
        
        # Split into train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        # Create 1:9 majority:minority class imbalance for training data
        class_0_idxs = np.where(y_train == 0)[0] # Majority class
        class_1_idxs = np.where(y_train == 1)[0] # Minority class
        new_class_0_size = len(class_1_idxs) // 9
        new_class_0_idxs = np.random.choice(class_0_idxs, new_class_0_size)
        new_idxs = list(new_class_0_idxs) + list(class_1_idxs)
        X_train = X_train[new_idxs]
        y_train = y_train[new_idxs]
        
        # Randomly flip a percentage of training labels
        n = len(y_train)
        y_train = np.array([abs(1 - y_train[i]) if i in random.sample(range(n), int(noise_level * n)) else y_train[i] for i in range(n)])

        if display:
            plt.scatter(X_train[:,0], X_train[:,1], c=['#516091' if y==1 else '#A9ECA2' for y in y_train])
            plt.savefig(display_path + '/covariate_shift/train_' + str(noise_level) + '_percent_noise.png')
            plt.close()
            plt.scatter(X[:,0], X[:,1], c=['#516091' if y_i==1 else '#A9ECA2' for y_i in y])
            plt.savefig(display_path + '/covariate_shift/train_' + str(noise_level) + '_percent_noise.png')
            plt.close()
        
        y_train = y_train.reshape(-1, 1)
        X_train = torch.tensor(X_train, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        
        y_test = y_test.reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
        
        return X_train, X_test, y_train, y_test