from abc import ABC, abstractmethod

"""Interface for risk functional classes."""
class RiskFunctionalInterface(ABC):

    @abstractmethod
    def forward(self, loss):
        """Forward pass of risk functional. Modifies a sequence of loss values."""
        raise NotImplementedError
