import torch
from torch.autograd import Variable
from .cvar_optimizer import CVaROptimizerInterface

"""CVaR implementation of "Distributionally robust losses for latent covariate mixtures" (Duchi et al. 2020)"""
class DROCovariate(CVaROptimizerInterface):
    """
        Parameters:
            p - parameter for DRO method
            eps - parameter for DRO method
            L - parameter for DRO method
            X - data which the norm of must be computed.
                This process takes a while, so a smaller subset of the training data may be used for an approximation.
            batch_size - batch size used for training
            optimizer - PyTorch optimizer for model parameters (e.g. SGD, Adam)
            eta_lr - learning rate for eta variable.
            beta_lr - learning rate for Beta variable.
    """
    def __init__(self, p, eps, L, X, batch_size, optimizer, eta_lr=1e-3, beta_lr=1e-3):
        super().__init__()
        self.p = p
        self.eps = eps
        self.L = L
        self.eta = Variable(torch.zeros(1), requires_grad=True)
        self.Beta = Variable(torch.ones(batch_size, batch_size), requires_grad=True)
        self.eta_lr = eta_lr
        self.beta_lr = beta_lr
        self.rows = torch.arange(0, batch_size)
        self.optimizer = optimizer

        c = torch.combinations(torch.arange(0, X.shape[0]), with_replacement=True)
        combined = torch.cat((c, c.flip(1)), dim=0).unique(dim=0)
        X_diff = torch.sub(X[combined][:,0,:], X[combined][:,1,:])
        self.X_norm = torch.linalg.norm(X_diff**(p-1), dim=1)

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.eta.grad.data.zero_()
        self.Beta.grad.data.zero_()

    def step(self, loss):
        diag_diff = self.Beta[self.rows] - torch.transpose(self.Beta, 0, 1)[self.rows]
        dro_loss = ((self.p - 1) * torch.mean(torch.relu(loss - torch.mean(diag_diff, dim=1) - self.eta))**self.p)**(1 / self.p)
        dro_loss += self.L**(self.p - 1) / (self.eps * self.bsz**2) * torch.sum(torch.mul(self.X_norms, self.Beta.flatten()))

        # Calculate gradient
        dro_loss.backward(retain_graph=True)

        # Gradient descent
        self.optimizer.step()

        # Update eta
        self.eta.data -= self.eta_lr * self.eta.grad.data

        # Update Beta
        self.Beta.data -= self.Beta_lr * self.Beta.grad.data
        self.Beta.data = torch.relu(self.Beta.data)