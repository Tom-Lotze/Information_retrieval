import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import torch

import time
sys.path.append('..')
sys.path.append('.')


class RankNet(nn.Module):
    """ Pairwise LTR model """

    def __init__(self, input_dim,  n_hidden=256,  output_dim=1):
        """
        Initialize model
        input_dim: dimensionality of document feature vector
        n_hidden: dimensionality of hidden layer
        n_outputs: output dimensionality
        """
        super(RankNet, self).__init__()

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_batch):
        """
        Forward pass
        - x: batch of document pairs B x (doc_i, doc_j)
        - out: batch of document scores: B x 1
        """
        # d_i = pairs[:, 0, :]
        # d_j = pairs[:, 1, :]

        h_i = self.relu(self.fc1(x_batch))
        s_i = self.relu(self.fc2(h_i))

        return s_i

    def evaluate_on_validation(self, data):
        """ evaluate on validation """
        valid_data = data.validation
        with torch.no_grad():
            valid_scores = self.forward(
                torch.Tensor(valid_data.feature_matrix))
            valid_scores = valid_scores.numpy().squeeze()
            results = evl.evaluate(valid_data, valid_scores)
        return results


class Loss_function(nn.Module):
    """ Pairwise LTR model """

    def __init__(self, gamma=1):
        """ Loss function init """
        super(Loss_function, self).__init__()
        self.gamma = gamma

    def forward(self, y_hat, y):
        """
        y_hat: 1d tensor of document scores
        """

        diff_mat = Variable(torch.sigmoid(y_hat.expand(
            y_hat.shape[0], y_hat.shape[0]).t() - y_hat), requires_grad=True)

        labels_mat = y.repeat(y.shape[0], 1).t() - y

        labels_mat[labels_mat > 0] = 1
        labels_mat[labels_mat == 0] = (1/2)
        labels_mat[labels_mat < 0] = 0

        # loss = (1/2) * (1 - labels_mat) * self.gamma * diff_mat + \
        # torch.log(1+torch.exp(-self.gamma * diff_mat))

        loss = nn.functional.binary_cross_entropy(diff_mat, labels_mat)

        return loss


def weights_init(model):
    """ Initialize weights """
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


def train(data):
    model = RankNet(data.num_features, 256, 1)
    model.apply(weights_init)

    n_epochs = 10

    train_dataset = dataset.ListDataSet(data.train)
    train_dl = DataLoader(train_dataset)

    # valid_dataset = dataset.ListDataSet(data.validation)
    # valid_dl = DataLoader(valid_dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    step = 0

    loss_function = Loss_function(gamma=1)

    for epoch in range(n_epochs):

        with tqdm(total=len(train_dl)) as t:
            t.set_description(f'Epoch: {epoch+1}/{n_epochs}')

            for step, (x_batch, y_batch) in enumerate(train_dl):

                optimizer.zero_grad()

                # ignore batch if only one doc (no pairs)
                if x_batch.shape[1] == 1:
                    continue

                # if step > 5:
                #     continue

                x_batch = x_batch.float().squeeze()
                # squeeze batch

                y_batch = y_batch.squeeze().float()

                scores = model(x_batch).squeeze()

                # define loss function
                loss = loss_function(scores, y_batch)

                if step % 100 == 0:
                    # model.eval()
                    valid_results = model.evaluate_on_validation(data)
                    # ndcg_10 = evl.ndcg_at_k(sorted_labels, 10)
                    t.set_postfix_str(
                        f'loss: {loss.mean().item():3f} ndcg@20: {valid_results["ndcg@20"][0]:3f}')

                # take mean to compare between batches
                # loss = loss.sum()

                # backward pass
                loss.backward(retain_graph=True)

                # optimizer step
                optimizer.step()

                t.update()


if __name__ == "__main__":
    # currently importing dataset here as pep8 would move it to the
    # top messing up the sys.path code
    import dataset
    import evaluate as evl

    # # load and read data
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    # # print data stats
    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' %
          data.validation.num_queries())
    print('Number of documents in validation set: %d' %
          data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())

    # train Model

    train(data)
