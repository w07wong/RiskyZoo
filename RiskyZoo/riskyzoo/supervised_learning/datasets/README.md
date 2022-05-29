# Datasets
We provide benchmark datasets for evaluating risk functionals.

## Datsets supported:
- 2D noisy label classification task
    - Returns tensors for train and test covariates and labels.
    - Generates 2 blobs of data with configurable amounts of noise.
    - One class receives 70% of all noise, the other receives 30%. 
- 2D covariate shift classification task
    - Returns tensors for train and test covariates and labels.
    - The training dataset contains a 1:9 ratio of class 1 to class 2 data
    - The test dataset follows the original data distribution
- Fairness regression task
- Label shift regression task
