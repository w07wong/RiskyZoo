# Risk functionals
Each risk functional takes as input a tensor of losses. Depending on the reduction specified, the forward pass returns either the mean or sum of the transformed losses, or when applicable, indvidual transformed values.

## Risk functionals supported:
- CVaR: ```cvar.py```
- Entropic risk: ```entropic_risk.py```
- Human-aligned risk: ```human_aligned_risk.py```
- Inverted CVaR: ```cvar.py```
- Mean-variance: ```mean_variance.py```
- Trimmed Risk: ```trimmed_risk.py```

## Adding new risk functionals
All risk functionals implement the RiskFunctionalInterface defined in ```risk_functional.py```