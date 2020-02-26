# This file implements a skip gram model (word2vec) using Pytorch

import numpy as np
import torch
from read_ap import *
import pickle as pkl
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from scipy.spatial.distance import cosine as cos_similarity


class Skipgram(nn.Module):
    def __init__(self, docs, vocab, counter, aggregation_function,
                 embedding_dim=300, window_size=5):
        """
        Initialization of the skip gram model
        Arguments:
            - docs: documents on which the embeddings are learned
            - output_dim: dimension of the embeddings that are learned
            - window_size: window size in the skip gram model

        """

        super(Skipgram, self).__init__()

        # set tunable parameters
        self.embedding_dim = embedding_dim
        self.window_size = 5
        self.nr_epochs = 20
        self.k = 10

        self.counter = counter

        # vocab needs to be made and filtered on infrequent words
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.aggr_function = aggregation_function

        self.target_embedding = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True)
        self.context_embedding = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True)

        nn.init.uniform_(self.target_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.context_embedding.weight, -0.1, 0.1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Perform a forward pass on the tuple of two one hot encodings"""

        # indices for positive and negative examples
        pos_examples = x[1]
        neg_examples = x[2]

        batch_size = len(pos_examples) + len(neg_examples)

        pos_t = [x[0]] * len(pos_examples)
        neg_t = [x[0]] * len(neg_examples)

        # target embedding (index the embeddings)
        pos_t_E = self.target_embedding(Variable(torch.Tensor(pos_t).long()))

        # positive embeddings
        pos_E = self.context_embedding(Variable(
            torch.Tensor(pos_examples).long()))

        pos_score = torch.sum(torch.mul(pos_t_E, pos_E), dim=1)
        pos_score = torch.log(self.sigmoid(pos_score)).squeeze()

        # pos_loss = Variable(torch.sum(pos_score, dim=1),
        # requires_grad=True).squeeze()

        neg_t_E = self.target_embedding(Variable(torch.Tensor(neg_t).long()))

        # similar as above
        # print(neg_examples)
        neg_E = self.context_embedding(
            Variable(torch.Tensor(neg_examples).long()))

        # neg_E = neg_E.view(neg_E.shape[0], 1, self.embedding_dim)
        # neg_score = self.sigmoid(cossim(target_E, neg_E))
        # print(f'neg_E size: {neg_E.size()}, target_E size :{target_E.size()}')
        neg_score = torch.sum(torch.bmm(neg_E,
                                        neg_t_E.unsqueeze(2)).squeeze(), dim=1)

        neg_score = torch.log(self.sigmoid(-neg_score)).squeeze

        # neg loss is just cosine similarity
        # neg_loss = Variable(torch.sum(neg_score, dim=0), requires_grad=True)

        return -(pos_score.sum() + neg_score.sum()) / batch_size

    def word_embedding(self, word):
        # inference model that returns an embedding for the word
        word_index = self.word_to_idx(word, self.vocab)
        return self.embedding(word_index)

    def doc_embedding(self, doc):
        # aggregate word embeddings
        pass

    def word_to_onehot(self, word):
        index = self.word_to_idx(word)
        onehot = torch.zeros(self.vocab_size)
        onehot[index] = 1

        return onehot

    def word_to_idx(self, word):
        """Returns the index of the word (string) in the vocabulary (dict)"""
        return self.vocab[word]

    def idx_to_word(idx, vocab):
        """
        Returns the word (string) corresponding to the index in the vocabulary
        (dict)
        """
        pass

    def get_neg_sample_pdf(self, counter):
        denominator = np.sum([np.power(counter[w], 3/4)
                              for w in self.vocab.keys()])

        sampling_list = [np.power(counter[word], 3/4)
                         for word in self.vocab.keys()] / denominator

        return sampling_list

    def neg_sample(self, counter, pdf):
        return np.random.choice(list(self.vocab.keys()), p=pdf, size=self.k)


def get_batches(model, docs):
    batches = []

    # denominator = np.sum([np.power(model.counter[w], 3/4)
    # for w in model.vocab.keys()])
    pdf = model.get_neg_sample_pdf(model.counter)

    top = int(model.window_size/2)

    for j, doc in enumerate(docs.values()):
        if j % 10 == 0:
            print(f'{j} docs processed')

        padded_doc = ["NULL"] * top + doc + ["NULL"] * top

        for i, target_word in enumerate(doc):

            if target_word not in model.vocab.keys():
                continue

            i += top

            window = padded_doc[i-top: i]+padded_doc[i+1: i+top+1]

            # instead of returning matrices of one hots, make batches
            # of indices
            t_indx = model.word_to_idx(target_word)
            pos_indx = [model.word_to_idx(
                c) for c in window if c != "NULL" and c in model.vocab.keys()]

            neg_indx = [model.word_to_idx(c)
                        for c in model.neg_sample(model.counter, pdf)]

            batches.append([t_indx, pos_indx, neg_indx])

    return batches


# training function for the embeddings


def train_skipgram(model, docs):

    np.random.seed(42)

    optimizer = optim.SparseAdam(model.parameters())

    BCE_loss = nn.BCELoss()
    batches = get_batches(model, docs)

    for epoch in range(model.nr_epochs):
        print(f"Epoch nr {epoch}")

        for step, X in enumerate(batches):
            optimizer.zero_grad()

            loss = model.forward(X)

            # print(f'Predictions: {predictions}')

            # loss = BCE_loss(predictions, torch.Tensor(y))

            if step % 1000 == 0:
                # print(f'Loss: {loss}')
                print(loss)

            loss.backward()
            optimizer.step()
