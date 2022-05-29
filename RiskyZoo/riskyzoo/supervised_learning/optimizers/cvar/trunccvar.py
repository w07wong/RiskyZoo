import torch
from torch.autograd import Variable
from .cvar_optimizer import CVaROptimizerInterface

"""Optimizer for TruncCVaR."""
class TruncCVaR(CVaROptimizerInterface):
    """
        Parameters:
            alpha - CVaR alpha
            optimizer - PyTorch optimizer for model parameters (e.g. SGD, Adam)
            eta_lr - learning rate for dual formulation of CVaR.
    """
    def __init__(self, alpha, optimizer, eta_lr=1e-3):
        super().__init__()
        self.alpha = alpha
        self.optimizer = optimizer
        self.eta = Variable(torch.zeros(1), requires_grad=True)
        self.eta_lr = eta_lr

    """
        Transforms the loss to the dual form of CVaR.
    """
    def _get_loss(self, loss):
        return (1 / self.alpha) * torch.mean(torch.relu(loss - self.eta)) + self.eta

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.eta.grad.data.zero_()

    def step(self, loss):
        # Get dual form of CVaR loss.
        dual_loss = self._get_loss(loss)
        # Compute gradients.
        dual_loss.backward(retain_graph=True)
        # Backpropagate using optimizer passed in.
        self.optimizer.step()
        # Update eta
        self.eta.data -= self.eta_lr * self.eta.grad.data
