from abc import ABC, abstractmethod

"""Interface for CVaR optimizers."""
"""
    Optimizers optimize for CVaR on top of an existing PyTorch optimizer.

    Each takes as input a PyTorch optimizer such as SGD to optimize model parameters.
    Then, the optimization method is modified to fit the CVaR optimization scheme.
    NOTE: You do not need to call loss.backward().
"""
class CVaROptimizerInterface(ABC):

    @abstractmethod
    def zero_grad(self):
        """Zeros out gradients of the computation graph."""

    @abstractmethod
    def step(self):
        """One step of the optimizer. Includes loss.backward() call."""
        raise NotImplementedError