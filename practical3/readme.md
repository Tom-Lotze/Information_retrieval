# Practical 3

This is the code base for practical 3, for the Information Retrieval 1 course @ UvA, Amsterdam.

## Requirements
One can use the environment.yml file.


## Usage
All files should be called from the main directory (practical3/)

### Pointwise LTR
- The model code can be found in the pointwise_ltr.py file. Running this file will train the model given the (optional) command line arguments, save the model and results, and plot the loss and ndcg over time.
- Tuning is done through the pointwise_tuning.sh, which will automatically train various models ad compute their results on the test set. The best configuration can be selected by running the process_jsons.py file, which will analyze all the results for the different models and return the best one. 
- To compute the distribution of the labels and predictions, make sure one model is run with the "save_pred" flag True, this ensures a pickle will be saved with the predictions on the test set (as predictions.pt). Then, running distribution_scores.py will compute the distributions of the predictions, test set labels and validation set labels and create plots that are saved in the distribution folder. 



### Pairwise LTR

### Listwise LTR
The model code is contained in ./listwise_ltr/listwise_ltr.py. Running the train.py file in the same folder will do (in the following order):

1. Hyper-parameter optimization
2. Training two models: the optimal configuration for nDCG and ERR, and save these to ./listwise_ltr/models. It will also produce a plot for the training progress in ./listwise_ltr/figures.
3. These final models are then used to evaluate on the test set. The metrics are stored in a pickle in ./listwise_ltr/results.
