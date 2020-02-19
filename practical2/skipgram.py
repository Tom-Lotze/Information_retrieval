# This file implements a skip gram model (word2vec) using Pytorch

import numpy as np
import torch
from read_ap import *
import pickle as pkl
import os
import torch.nn as nn
import torch.optim as optim
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
        self.nr_epochs = 5
        self.k = 10

        self.counter = counter

        # vocab needs to be made and filtered on infrequent words
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.aggr_function = aggregation_function

        self.target_fc = nn.Linear(self.vocab_size, self.embedding_dim)
        self.context_fc = nn.Linear(self.vocab_size, self.embedding_dim)

        self.target_embedding = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True)
        self.context_embedding = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Perform a forward pass on the tuple of two one hot encodings"""

        cossim = nn.CosineSimilarity(dim=1)

        # target index
        t_w = x[0]

        # indices for positive and negative examples
        pos_examples = x[1]
        neg_examples = x[2]

        # t_w = x[0, :].view(1, self.vocab_size)

        # target embedding (index the embeddings)
        target_E = self.target_embedding(torch.Tensor([t_w]).long())

        # positive embeddings
        pos_E = self.context_embedding(torch.Tensor(pos_examples).long())
        # cosine similarity
        pos_score = self.sigmoid(cossim(target_E, pos_E))

        # loss of positive examples, should score 1,
        # loss is difference between 1 and actual score
        pos_loss = torch.sum(1 - pos_score, dim=0)

        # similar as above
        # print(neg_examples)
        neg_E = self.context_embedding(torch.Tensor(neg_examples).long())
        neg_score = self.sigmoid(cossim(target_E, neg_E))

        # neg loss is just cosine similarity
        neg_loss = torch.sum(neg_score)

        # context_E = self.context_fc(c_w)

        # print(f'target_e is {target_E.size()}')
        # print(f'cotext_e is {context_E.size()}')

        # cos = cossim(target_E, context_E)

        # return self.sigmoid(cos)

        return pos_loss + neg_loss

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

            window = padded_doc[i-top:i]+padded_doc[i+1:i+top+1]

            # instead of returning matrices of one hots, make batches of indices
            t_indx = model.word_to_idx(target_word)
            pos_indx = [model.word_to_idx(
                c) for c in window if c != "NULL" and c in model.vocab.keys()]

            neg_indx = [model.word_to_idx(c)
                        for c in model.neg_sample(model.counter, pdf)]

            # pos_tuples = [(model.word_to_onehot(target_word),
            #                model.word_to_onehot(c)) for c in window if c != "NULL" and c in model.vocab.keys()]
            # negative samples
            # neg_tuples=[(model.word_to_onehot(target_word), model.word_to_onehot(
            # c)) for c in model.neg_sample(model.counter, pdf)]

            # all_tuples = pos_tuples + neg_tuples
            # all_labels = [1] * len(pos_tuples) + [0] * len(neg_tuples)

            # batch_size=len(pos_tuples) + model.k

            # batch_x=torch.Tensor(batch_size + 1, model.vocab_size)

            # batch_pos=torch.Tensor(len(pos_tuples), model.vocab_size)
            # batch_neg=torch.Tensor(len(neg_tuples), model.vocab_size)

            # for i, tup in enumerate(pos_tuples):
            #     batch_pos[i, :]=tup[1]

            # for i, tup in enumerate(neg_tuples):
            #     batch_neg[i, :]=tup[1]

            # batch_target=model.word_to_onehot(target_word)

            # batch_labels = torch.zeros(batch_size)
            # batch_labels[:len(pos_tuples)] = 1

            # for (tup, label) in zip(all_tuples, all_labels):
            # batches.append([batch_target, batch_pos, batch_neg])
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

            if step % 100 == 0:
                print(f'Loss: {loss}')

            loss.backward()
            optimizer.step()
