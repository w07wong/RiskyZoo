import torch
import torch.nn as nn
from .risk_functional import RiskFunctionalInterface

class EntropicRisk(RiskFunctionalInterface, nn.Module):
    """
        Parameters:
            t - entropic risk tilt parameter, can be negative, positive, or 0
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, t=10, reduction='mean'):
        super().__init__()
        self.t = t
        self.reduction = reduction
    
    def forward(self, loss):
        # t = 0 should return ERM
        if self.t == 0:
            return torch.mean(loss)
        
        if self.reduction == 'mean':
            return (1 / self.t) * torch.log(torch.mean(torch.exp(self.t * loss)))
        else:
            raise Exception('Only mean reduction type supported.')