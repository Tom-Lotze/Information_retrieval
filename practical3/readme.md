# Practical 3

This is the code base for practical 3, for the Information Retrieval 1 course @ UvA, Amsterdam.

## Requirements
PyTorch, Matplotlib, numpy, tqdm


## Usage

### PointWise LTR

### PairWise LTR

### ListWise LTR
The model code is contained in ./listwise_ltr/listwise_ltr.py. Running the train.py file in the same folder will do (in the following order):

1. Hyper-parameter optimization
2. Training two models: the optimal configuration for nDCG and ERR, and save these to ./listwise_ltr/models. It will also produce a plot for the training progress in ./listwise_ltr/figures.
3. These final models are then used to evaluate on the test set. The metrics are stored in a pickle in ./listwise_ltr/results.
