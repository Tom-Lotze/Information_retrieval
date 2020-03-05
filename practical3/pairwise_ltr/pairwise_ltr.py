import sys
import numpy as np
import torch.nn as nn
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

    def forward(self, pair_batch):
        """
        Forward pass
        - x: batch of document pairs B x (doc_i, doc_j)
        - out: batch of document scores: B x 1
        """

        d_i = torch.Tensor([doc_pair[0] for doc_pair in pair_batch])
        d_j = torch.Tensor([doc_pair[1] for doc_pair in pair_batch])

        h_i = self.relu(self.fc1(d_i))
        s_i = self.fc2(h_i)

        h_j = self.relu(self.fc1(d_j))
        s_j = self.fc2(h_j)

        diff = s_i - s_j

        return diff


def weights_init(model):
    """ Initialize weights """
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


def train(data):
    model = RankNet(data.num_features, 256, 1)
    model.apply(weights_init)

    labels = data.train.label_vector
    all_docs = data.train.feature_matrix

    batches = []

    print(f'Creating training pairs per query')
    for q_id in range(data.train.num_queries())[1:10000]:
        # print(f'query_id: {q_id}')
        s_i, e_i = data.train.query_range(q_id)

        q_r = np.arange(s_i, e_i)
        # print(f' query_range = {q_r}')
        # labels = data.train.query_labels(q_id)

        query_doc_pairs = [(i, j) for i in q_r for j in q_r if i != j]

        doc_pair_features = [(all_docs[i], all_docs[j])
                             for i, j in query_doc_pairs]

        query_doc_pairs_labels = [(labels[i], labels[j])
                                  for i, j in query_doc_pairs]

        true_pair_scores = [pair_score(i, j)
                            for i, j in query_doc_pairs_labels]

        x = doc_pair_features
        y = true_pair_scores

        batches.append((x, y))
    print(f'Created batches. Start training')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # see section 5

    for x_batch, y_batch in batches:
        optimizer.zero_grad()

        diff = model(x_batch)

        S_ij = torch.Tensor(y_batch)

        # define loss function
        # sigma = 1
        loss = (1/2) * (1 - S_ij) * diff + \
            torch.log(1 + torch.exp(-1 * diff))

        loss = torch.mean(loss)
        print(loss.item())
        loss.backward()
        optimizer.step()


def pair_score(s_i, s_j):
    if s_i > s_j:
        return 1
    elif s_i < s_j:
        return -1
    elif s_i == s_j:
        return 0


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
    # toy data:

    train(data)
