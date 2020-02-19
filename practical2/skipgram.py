
# This file implements a skip gram model (word2vec) using Pytorch

import numpy as np
import torch
from read_ap import *
import pickle as pkl 
import os
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cosine as cos_similarity



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

        super(MLP, self).__init__()

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

        self.target_embedding = nn.Embedding(self.vocab_size, self.output_dim)
        self.context_embedding = nn.Embedding(self.vocab_size, self.output_dim)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """Perform a forward pass on the tuple of two one hot encodings"""
        (target_word, context_word) = x
        target_E = self.target_fc(target_word)
        context_E = self.context_fc(context_word)

        cos = cos_similarity(target_E, context_E)

        return self.sigmoid(cos)


    def word_embedding(self, word):
        # inference model that returns an embedding for the word
        word_index = self.word_to_idx(word, self.vocab)
        return self.embedding(word_index)


    def doc_embedding(self, doc):
        #aggregate word embeddings
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
        denominator = np.sum([np.pow(f, 3/4) for f in counter.values()])
        
        sampling_list = [np.pow(counter[word], 3/4) for word in self.vocab.keys()] / denominator

        print(f"Sum of pdf {sum(sampling_list)}")

        return sampling_list


    def neg_sample(self, counter):
        return np.random.choice(self.vocab.keys(), p=get_neg_sample_pdf(counter), size=self.k)





# training function for the embeddings
def train_skipgram(model, docs):
        
        top = (model.window_size-1)/2

        np.random.seed(42)

        optimizer = optim.SparseAdam(SKIP.parameters())

        for epoch in model.nr_epochs:
            print(f"Epoch nr {epoch}")

        
            for doc in docs:
                doc_len = len(doc)
                padded_doc = ["NULL"]*top + doc + ["NULL"]*top
                for i, target_word in enumerate(doc):
                    i += top
                    window = padded_doc[i-top:i]+padded_doc[i+1:i+top+1]
                    pos_tuples = [(model.word_to_onehot(target_word), model.word_to_onehot(c)) for c in window]

                    # negative samples
                    neg_tuples = [(model.word_to_onehot(target_word), model.word_to_onehot(c)) for c in neg_sample(model.counter)]

                    all_tuples = pos_tuples + neg_tuples
                    all_labels = [1] * len(pos_tuples) + [0] * len(neg_tuples)

                    for (tup, label) in zip(all_tuples, all_labels):
                        optimizer.zero_grad()
                        predictions = model.forward(tup)

                        loss = nn.CrossEntropyLoss(predictions, label)

                        loss.backward()
                        optimizer.step()









    




