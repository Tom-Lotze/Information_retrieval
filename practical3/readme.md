# Practical 3

This is the code base for practical 3, for the Information Retrieval 1 course @ UvA, Amsterdam.

## Requirements
One can use the environment.yml file.


## Usage
All files should be called from the main directory (practical3/). For each model there exists a seperate folder, indicated in brackets in the headings. 

### Pointwise LTR (pointwise_ltr)
- The model code can be found in the pointwise_ltr.py file. Running this file will train the model given the (optional) command line arguments, save the model and results, and plot the loss and ndcg over time.
- Tuning is done through the pointwise_tuning.sh, which will automatically train various models ad compute their results on the test set. The best configuration can be selected by running the process_jsons.py file, which will analyze all the results for the different models and return the best one. 
- To compute the distribution of the labels and predictions, make sure one model is run with the "save_pred" flag True, this ensures a pickle will be saved with the predictions on the test set (as predictions.pt). Then, running distribution_scores.py will compute the distributions of the predictions, test set labels and validation set labels and create plots that are saved in the distribution folder. 

### Pairwise LTR
#### Default Ranknet (pairwise_ltr)
- The model code is in the pairwise_ltr.py file. Running this file will train the model using the (optional) command line parameters and if needed, saves the models, results and plots. 
- Tuning is done through the pairwise_tuning.sh script. The best model is then found by running process_jsons.py, which returns the best model by analyzing the jsons produced in the gridsearch. 


#### Sped-up Ranknet (pairwise_ltr_sped_up)
- The model code is in the pairwise_ltr_sped_up.py file. Running this file will train the model using the (optional) command line parameters and if needed, saves the models, results and plots. 
- Tuning is done through the sped_up_tuning.sh script. The best model is then found by running process_jsons.py, which returns the best model by analyzing the jsons produced in the gridsearch. 

### Listwise LTR (listwise_ltr)
- The code to train the LambdaRank model is in listwise_ltr.py. Running this file will train the model using the command line arguments.
- Tuning the model can be done by running listwise_tuning.sh for the NDCG LambdaRank or listwise_tuning_err.sh for the ERR LambdaRank.


# Command line arguments:

## General:
- --learning rate - learning rate for optimizer
- --max_epochs - maximum number of epochs to train
- --save - boolean to save the trained model
- --plot - boolean to save plots of NDCG (and ARR for pairwise_ltr.py) during training
- --valid_each - frequency to evaluate on validation set (frequency in batches)
- --early_stopping_threshold - threshold for early stopping
- --save_pred - boolean to save predictions on test set

## Listwise specific
- --metric - to indicate which metric to use during training. 'NDCG' or 'ERR'.

## Pointwise specific
- --batch_size - batch size for training
