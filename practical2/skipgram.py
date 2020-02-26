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
        self.nr_epochs = 2000
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
        nn.init.uniform_(self.context_embedding.weight, 0, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_u, pos_v, neg_v):
        """Perform a forward pass on the tuple of two one hot encodings"""

        # indices for positive and negative examples
        batch_size = len(pos_u)
        pos_u = Variable(torch.Tensor(pos_u).long())
        pos_v = Variable(torch.Tensor(pos_v).long())
        neg_v = Variable(torch.Tensor(neg_v).long())

        # target embedding (index the embeddings)
        pos_t_E = self.target_embedding(pos_u)

        # positive embeddings
        pos_E = self.context_embedding(pos_v)

        pos_score = torch.sum(torch.mul(pos_t_E, pos_E).squeeze(), dim=1)
        pos_score = torch.log(self.sigmoid(pos_score))

        neg_E = self.context_embedding(neg_v)

        neg_score = torch.bmm(neg_E, pos_t_E.unsqueeze(2))
        neg_score = torch.sum(neg_score, dim=1)

        neg_score = torch.log(self.sigmoid(-neg_score)).squeeze()

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

    def neg_sample(self, counter, pdf, k):
        return np.random.choice(list(self.vocab.keys()), p=pdf, size=k)


def get_batches(model, docs, batch_size):
    ''' generate batches '''

    pdf = model.get_neg_sample_pdf(model.counter)

    top = int(model.window_size/2)

    pos_batch = []
    neg_batch = []

    for j, doc in enumerate(docs.values()):
        # if j % 10 == 0:
        # print(f'{j} docs processed')

        padded_doc = ["NULL"] * top + doc + ["NULL"] * top

        for i, target_word in enumerate(doc):

            if target_word not in model.vocab.keys():
                continue

            i += top

            window = padded_doc[i-top: i]+padded_doc[i+1: i+top+1]

            pos_pairs = [(model.word_to_idx(target_word), model.word_to_idx(
                c)) for c in window if c != "NULL" and c in model.vocab.keys()]

            pos_batch += pos_pairs

            if len(pos_batch) == batch_size:
                for pos_x in pos_batch:
                    neg_batch.append([model.word_to_idx(c)
                                      for c in model.neg_sample(model.counter,
                                                                pdf, model.k)])
                yield (pos_batch, neg_batch)
                neg_batch = []
                pos_batch = []
                pos_pairs = []
                # batches.append([t_indx, pos_indx, neg_indx])

                # return batches

                # training function for the embeddings


def train_skipgram(model, docs):

    np.random.seed(42)

    optimizer = optim.SparseAdam(model.parameters())

    # batches = get_batches(model, docs)

    batch_size = 50

    for epoch in range(model.nr_epochs):
        print(f"Epoch nr {epoch}")

        for step, (pos_batch, neg_batch) in enumerate(get_batches(model, docs, batch_size)):
            optimizer.zero_grad()

            pos_u = [x[0] for x in pos_batch]
            pos_v = [x[1] for x in pos_batch]
            neg_v = neg_batch

            loss = model.forward(pos_u, pos_v, neg_v)

            if step % 1000 == 0:
                # print(f'Loss: {loss}')
                print(loss)

            loss.backward()
            optimizer.step()
