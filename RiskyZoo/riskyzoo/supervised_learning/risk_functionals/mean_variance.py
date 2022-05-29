import torch
import torch.nn as nn
from .risk_functional import RiskFunctionalInterface

class MeanVariance(RiskFunctionalInterface, nn.Module):
    """
        Parameters:
            c - variance penalty
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, c=0.1, reduction='mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction
    
    def forward(self, loss):
        var, mean = torch.var_mean(loss, unbiased=False)
        
        if self.reduction == 'mean':
            return mean + torch.mul(var, self.c)
        elif self.reduction == 'sum':
            return torch.sum(loss + torch.mul(var, self.c))
        elif self.reduction == 'none':
            return loss + torch.mul(var, self.c)
        else:
            raise Exception('Only mean, sum, none reduction types supported.')