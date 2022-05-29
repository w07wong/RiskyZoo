import torch
from torch.autograd import Variable
from .cvar_optimizer import CVaROptimizerInterface

"""CVaR implementation of "Learning models with uniform performance via distributionally robust optimization" (Duchi et al. 2018)"""
class DROJoint(CVaROptimizerInterface):
    """
        Parameters:
            k - parameter for DRO method
            rho - parameter for DRO method
            optimizer - PyTorch optimizer for model parameters (e.g. SGD, Adam)
            eta_lr - learning rate for eta variable.
    """
    def __init__(self, k, rho, optimizer, eta_lr=1e-3):
        super().__init__()
        self.k_star = k / (k - 1)
        self.c_k = (1 + k * (k - 1) * rho) ** (1 / k)
        self.optimizer = optimizer
        self.eta_lr = eta_lr
        self.eta = Variable(torch.zeros(1), requires_grad=True)

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.eta.grad.data.zero_()

    def step(self, loss):
        dro_loss = self.c_k * torch.mean(torch.relu((loss - self.eta) ** self.k_star)) ** (1 / self.k_star) + self.eta
        # Calculate gradients
        dro_loss.backward(retain_graph=True)
        # Gradient descent
        self.optimizer.step()
        # Update eta
        self.eta.data -= self.eta_lr * self.eta.grad.data