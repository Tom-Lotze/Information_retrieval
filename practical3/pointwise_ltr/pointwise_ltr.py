import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append('..')
sys.path.append(".")

import dataset
#import ranking
#import evaluate


class Pointwise(nn.Module):
    """


    """
    def __init__(self, n_inputs, n_hidden, n_outputs=1):
        """
        Model that takes an input feature vector x and tries to predict the relevance score
        Input arguments:
        - n_inputs: the dimensionality of the feature vector
        - n_hidden: list of dimensions for the hidden layers
        - n_outputs: output dimensionality
        """
        super(Pointwise, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.relu = nn.ReLU()

        # add linear layers
        layer_list = [nn.Linear(n_inputs, n_hidden[0])]
        if len(n_hidden) > 1:
            for i, n_hid in enumerate(n_hidden[1:]):
                layer_list.append(nn.Linear(n_hidden[i-1], n_hid))
        layer_list.append(nn.Linear(n_hidden[-1], n_outputs))

        self.layers = nn.ModuleList(layer_list)

        print(self.layers)
        


    def forward(self, x):
        for layer in self.layers():
            x = layer(x)
            x = self.relu(x)


        return x



    def evaluate_on_validation(self, x_valid, y_valid):
        pass

    def evaluate_on_test(self, x_test, y_test):
        pass





# helper functions
def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


# train function
def train(data):
    model = Pointwise(data.num_features, [512, 256, 128, 64, 8], 1)
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    x_train, y_train = data.train.feature_matrix, data.train.label_vector

    print(f"xtrain shape: {x_train.shape}, labels: {y_train.shape}")






if __name__ == "__main__":
    # import the data
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()


    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' % data.validation.num_queries())
    print('Number of documents in validation set: %d' % data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())

    
    train(data)



