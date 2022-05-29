import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

"""Generates training and testing data for noisy label classification task. Returns tensors of data."""
class NoisyLabelsClassification():
    def _flip_labels(self, y, class1_noise, class2_noise):
        # Flip class 1 labels from 1 to 0
        class1 = y[np.where(y == 1)]
        class1_size = len(class1)
        y[np.where(y == 1)] = np.array([abs(1 - class1[i]) if i in random.sample(range(class1_size), int(class1_noise * class1_size)) else class1[i] for i in range(class1_size)])

        # Flip class 2 labels from 0 to 1
        class2 = y[np.where(y == 0)]
        class2_size = len(class2)
        y[np.where(y == 0)] = np.array([abs(1 - class2[i]) if i in random.sample(range(class2_size), int(class2_noise * class2_size)) else class2[i] for i in range(class2_size)])

        return y

    """Method to call to generate data."""
    def generate_data(self, n_samples_1, n_samples_2, noise_level, test_shift=False, seed=15, display=False, display_path=''):
        centers = [[0.0, 0.0], [1, 1.]]
        clusters_std = [0.4, 0.4]
        X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                        centers=centers,
                        cluster_std=clusters_std,
                        random_state = 15,
                        shuffle=False)
        if test_shift:
            # Randomly flip a percentage of test labels
            y = self.flip_labels(y, noise_level * 0.7, noise_level * 0.3)
            
        # Split into train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        if not test_shift:
            # Randomly flip a percentage of training labels
            y_train = self.flip_labels(y_train, noise_level * 0.7, noise_level * 0.3)
        
        if display:
            plt.scatter(X_train[:,0], X_train[:,1], c=['#516091' if y==1 else '#A9ECA2' for y in y_train])
            plt.savefig(display_path + '/noisy_labels/train_' + str(noise_level) + '_percent_noise.png')
            plt.close()
            plt.scatter(X_test[:,0], X_test[:,1], c=['#516091' if y==1 else '#A9ECA2' for y in y_test])
            plt.savefig(display_path + 'noisy_labels/test_' + str(noise_level) + '_percent_noise.png')
            plt.close()
        
        y_train = y_train.reshape(-1, 1)
        X_train = torch.tensor(X_train, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        
        y_test = y_test.reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
        
        return X_train, X_test, y_train, y_test