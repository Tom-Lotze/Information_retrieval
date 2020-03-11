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

    def forward(self, x):
        """
        Forward pass
        - x: batch of document pairs B x (doc_i, doc_j)
        - out: batch of document scores: B x 1
        """

        h_i = self.relu(self.fc1(x))
        s_i = self.sigmoid(self.fc2(h_i))

        return s_i


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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    step = 0

    for epoch in range(n_epochs):

        with tqdm(total=len(train_dl)) as t:
            t.set_description(f'Epoch: {epoch+1}/{n_epochs}')
            for step, (x_batch, y_batch) in enumerate(train_dl):
                optimizer.zero_grad()

                # ignore batch if only one doc (no pairs)
                if x_batch.shape[1] == 1:
                    continue

                # squeeze batch
                y_batch = y_batch.squeeze()

                # compute scores per doc
                y_hat = model(x_batch.float()).squeeze()

                # compute score diffs per doc pair
                y_hat_diffs = Variable(torch.Tensor([(s_i - s_j)
                                                     for i, s_i in enumerate(y_hat)
                                                     for j, s_j in enumerate(y_hat)
                                                     if i != j]), requires_grad=True)

                # label pairs
                y_pairs = torch.Tensor([(s_i, s_j)
                                        for i, s_i in enumerate(y_batch)
                                        for j, s_j in enumerate(y_batch)
                                        if i != j])

                # compute real diffs
                S_ij = torch.zeros_like(y_hat_diffs)
                for s, (i, j) in enumerate(y_pairs):
                    if i > j:
                        S_ij[s] = 1
                    elif i == j:
                        S_ij[s] = 0
                    elif i < j:
                        S_ij[s] = -1

                # define loss function
                # sigma = 1
                loss = (1/2) * (1 - S_ij) * y_hat_diffs + \
                    torch.log(1 + torch.exp(-1 * y_hat_diffs))

                # take mean to compare between batches
                loss = torch.mean(loss)

                # backward pass
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    t.set_postfix_str(f'loss: {loss.item()}')
                t.update()


if __name__ == "__main__":
    # currently importing dataset here as pep8 would move it to the
    # top messing up the sys.path code
    import dataset

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
