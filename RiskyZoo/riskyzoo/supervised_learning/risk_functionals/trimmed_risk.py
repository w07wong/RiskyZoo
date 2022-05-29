import imp
import torch
import torch.nn as nn
from risk_functional import RiskFunctionalInterface

"""Class for trimmed risk."""
class TrimmedRisk(RiskFunctionalInterface, nn.Module):
    """
        Parameters:
            alpha - quantile of tails to ignore
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, alpha=0.05, reduction='mean'):
        super().__init__()
        assert alpha >= 0 and alpha <= 0.5, 'alpha must be in [0, 0.5]'
        self.alpha = alpha
        self.reduction = reduction
    
    """Gets losses not on the tail ends of the distribution."""
    def _get_untrimmed_losses(self, loss):
        sorted_indices = torch.argsort(loss, dim=0, descending=False)
        empirical_cdf = torch.argsort(sorted_indices) / len(loss)
        return ((empirical_cdf >= self.alpha) & (empirical_cdf <= 1 - self.alpha)).nonzero().squeeze()
    
    def forward(self, loss):
        untrimmed_losses = self._get_untrimmed_losses(loss)
        
        if self.reduction == 'mean':
            return torch.mean(torch.index_select(loss, 0, untrimmed_losses))
        elif self.reduction == 'sum':
            return torch.sum(torch.index_select(loss, 0, untrimmed_losses))
        elif self.reduction == 'none':
            return torch.index_select(loss, 0, untrimmed_losses)
        else:
            raise Exception('Only mean, sum, none reduction types supported.')