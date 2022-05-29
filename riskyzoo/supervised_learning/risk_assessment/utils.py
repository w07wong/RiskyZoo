import numpy as np
import matplotlib.pyplot as plt
import os
import torch

"""Plots the loss distribution to a file."""
def plot_loss_distribution(losses, path, fname, loss_name):
    plt.figure(figsize=(10,6))
    
    weights = np.ones_like(losses) / len(losses)
    plt.hist(x=losses, color='#272f4a',
                            alpha=1, weights=weights)
    
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(loss_name)
    plt.ylabel('Percent of Samples')

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + fname + '.png')
    
    plt.close()


"""Evaluates a model under multiple risk functionals and returns risk assessment."""
"""
    Parameters:
        losses - list of loss for each data point
        risk_functionals - dictionary mapping from risk functional name to risk functional object
    
    Example risk_functionals dictionary:
    eval_risk_functionals = {
        'Expected Value': nn.CrossEntropyLoss(),
        'CVaR-0.3': cvar.CVaR(a=0.3),
        'Entropic Risk': entropic_risk.EntropicRisk(t=-0.5),
        'Human-Aligned Risk': human_aligned_risk.HumanAlignedRisk(a=0.4, b=0.8),
        'Inverted CVaR-0.7': cvar.CVaR(a=0.7, inverted=True),
        'Mean-Variance': mean_variance.MeanVariance(c=1),
        'Trimmed Risk-0.3': trimmed_risk.TrimmedRisk(a=0.3),
    }
"""
def eval_under_risk_functionals(losses, risk_functionals):
    metrics = dict()
    for risk_functional in risk_functionals.keys():
        if risk_functional != 'Expected Value':
            metrics[risk_functional] = risk_functionals[risk_functional](losses)
        else:
            metrics[risk_functional] = torch.mean(losses)
