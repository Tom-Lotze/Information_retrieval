# distribution_scores.py
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
from collections import Counter
import pickle as pkl

from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('..')
sys.path.append(".")
import dataset
#import ranking
import evaluate as evl


# distribution of actual scores on validation and test set
def compute_actual_dist(data_set):
    labels = data_set.label_vector
    count = Counter(labels)

    return count

def compute_pred_dist():
    predictions = pkl.load(open("pointwise_LTR/predictions.pt", "r"))

    count = Counter(predictions)

    return count


def plot(counter, title):
    plt.figure()
    plt.bar(counter.keys(), counter.values())
    plt.title(title)
    plt.xlabel("Relevance class")
    plt.ylabel("Frequency")


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    # create instances specific for pointwise model
    data.train = dataset.Pointwise_fold(data.train)
    data.validation = dataset.Pointwise_fold(data.validation)
    data.test = dataset.Pointwise_fold(data.test)

    os.makedirs("pointwise_ltr/distribution", exist_ok=True)

    # distribution of labels
    actual_dist_valid = compute_actual_dist(data.validation)
    actual_dist_test = compute_actual_dist(data.test)

    # distribution of predictions
    pred_dist = compute_pred_dist()

    plot(actual_dist_test, "Test set")
    plot(actual_dist_valid, "Validation set")


    plt.show()
