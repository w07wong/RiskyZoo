# CVaR Optimizers
Implementations of CVaR optimziers can be found here. These optimizers are not standard PyTorch optimizers and instead provide a wrapper on top of the traditional optimizers (e.g. torch.optim.SGD).

## Usage
To usage these optimizers, you must pass in an optimzier you would like to use for model parameters such as the PyTorch SGD or Adam. Then, follow the format below.

```
...

    # Standard optimizer & scheduler definitions to begin.
    optimizer = torch.optim.SGD(...)
    scheduler = torch.optim.lr_scheduler.ANY_SCHEDULER(optimizer...) # You may use a learning rate scheduler.

    # Then, create a CVaR optimizer object.
    from riskyzoo.supervised_learning.optimizers.cvar.trunccvar import TruncCVaR
    cvar_optimizer = TruncCVaR(alpha=0.2, optimizer=optimizer)

    # Define loss function.
    # Note that this is a PyTorch loss like torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), etc.
    # REQUIREMENT: reduction must be set to 'none'.
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Training:
    for i in range(epochs):
        cvar_optimizer.zero_grad()

        pred = model(X_train)
        loss = criterion(pred, ground_truth)

        cvar_optimizer.step(loss)

...

```

## CVaR Optimizers Supported
- Dual form of CVaR: ```trunccvar.py```
- Adaptive sampling from "Adaptive Sampling for Stochastic Risk-Averse Learning" (Curi et al. 2020): ```adavar.py```
    - Code is source from https://github.com/sebascuri/adacvar
- SoftCVaR
- "Distributionally robust losses for latent covariate mixtures" (Duchi et al. 2020): ```dro_covariate.py```
- "Learning models with uniform performance via distributionally robust optimization" (Duchi et al. 2018): ```dro_joint.py```

## Adding new optimizers
All cvar risk functionals implement the CVaROptimizerInterface defined in ```cvar_optimizer.py```
