from utils.adacvar.adacvar.util.cvar import CVaR
from utils.adacvar.adacvar.util.adaptive_algorithm import Exp3Sampler
from .cvar_optimizer import CVaROptimizerInterface
import numpy as np
import torch
from torch.utils.data import DataLoader

"""Code sourced from "Adaptive Sampling for Stochastic Risk-Averse Learning" (Curi et al. 2020): https://github.com/sebascuri/adacvar"""
class AdaVaR(CVaROptimizerInterface):
    """
        Parameters:
            cvar - CVaR alpha
            batch_size - batch_size used for training
            dataset - PyTorch dataset object containing training data.
            optimizer - PyTorch optimizer for model parameters (e.g. SGD, Adam)
    """
    def __init__(self, alpha, batch_size, dataset, optimizer):
        super().__init__()
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.exp3 = Exp3Sampler(
            batch_size=batch_size,
            num_actions=self.dataset_size,
            size=int(np.ceil(alpha * self.dataset_size)),
            eta=0,
            gamma=0,
            beta=0,
            eps=1e-16,
            iid_batch=True,
        )
        self.loader = DataLoader(dataset, batch_sampler=self.exp3)
        self.adaptive_algorithm = self.loader.batch_sampler
        self.optimizer = optimizer
        self.cvar = CVaR(alpha=1, learning_rate=0)

    """Retursn training data loader for adaptive sampling algorithm. You must use this data loader."""
    def get_loader(self):
        return self.loader

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.cvar.zero_grad()

    """
        Parameters:
            losses - must be individual losses for each data point.

        Usage example:
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            adavar_optimzer.zero_grad()
            criterion = torch.nn.CrossEntropyLoss(reduction='none') #NOTE: reduction is none.
            output = model(X)
            loss = criterion(output, ground_truth_labels)
            adavar_optimizer.step(loss)

    """ 
    def step(self, losses, idx):
        # batch sampler (not cyclic sampler line 93 adacvar.util.train.py)
        weights = 1.0
        probabilities = self.adaptive_algorithm.probabilities

        # Feedback loss to sampler.
        self.adaptive_algorithm.update(
            1 - np.clip(losses.cpu().detach().numpy(), 0, 1), idx, probabilities
        )

        # Calculate CVaR and reduce to mean.
        cvar_loss = (torch.tensor(weights).float().to('cpu') * self.cvar(losses)).mean()

        # Compute gradietns and backpropagate.
        cvar_loss.backward()
        self.optimizer.step()
        self.cvar.step()

        # Renormalize sampler
        self.adaptive_algorithm.normalize()
