# Optimizers
Specialized optimizers for risk functionals. Currently, only optimizers for CVaR are studied. Refer to the cvar repository for usage details. For the naive gradient based optimization, simply use a risk functional from riskyzoo.supervised_learning.risk_functionals after computing the loss with a PyTorch loss like follows:

```
...

criterion = torch.nn.CrossEntropyLoss(reduction='none') # You must have reduction set to none.
risk_functional = riskyzoo.supervised_learning.risk_functionals.CVaR(a=0.2)

pred = model(X)
loss = criterion(pred, ground_truth)
loss = risk_functional(loss)

loss.backward()

...

```