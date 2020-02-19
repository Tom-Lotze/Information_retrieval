
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
    def __init__(self, docs, vocab_size, aggregation_function,
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
        
        # vocab needs to be made and filtered on infrequent words
        self.vocab = 0
        self.vocab_size = 0      
        
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



    def word_embedding(self, word, vocab=self.vocab):
        # inference model that returns an embedding for the word
        word_index = self.word_to_idx(word, vocab)
        return self.embedding(word_index)



    def doc_embedding(self, doc):
        #aggregate word embeddings
        pass



    def word_to_idx(self, word, vocab=self.vocab):
        """Returns the index of the word (string) in the vocabulary (dict)"""
        return vocab[word]


    def idx_to_word(idx, vocab):
        """
        Returns the word (string) corresponding to the index in the vocabulary
        (dict)
        """
        pass




def train_skipgram(docs):
    nr_epochs = 5

    np.random.seed(42)

    SKIP = Skipgram()
    optimizer = optim.SparseAdam(SKIP.parameters())

    for epoch in nr_epochs:
        optimizer.zero_grad()







    




