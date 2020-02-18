
# This file implements a skip gram model (word2vec) using Pytorch

import numpy as np
import torch
from read_ap import *
import pickle as pkl 
import os
import torch.nn as nn
import torch.optim as optim




class Skipgram(nn.Module):
    def __init__(self, docs, vocab_size, embedding_dim=300, window_size=5):
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
        self.vocab_size = 0        
        
        self.fc1 = nn.Linear(self.vocab_size, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, self.vocab_size)

        self.embedding = nn.Embedding(self.vocab_size, self.output_dim)


    def forward(self, x):
        """Perform a forward pass on the one-hot encoding of a word"""
        for l in self.layers():
            x = l(x)

        return x



    def word_embedding(self, word):
        # inference model that returns an embedding for the word
        pass
        #nn.Embedding



    def doc_embedding(self, doc):
        #aggregate word embeddings
        pass



def train_skipgram(docs):

    SKIP = Skipgram()
    optimizer = optim.SparseAdam(SKIP.parameters())




if __name__ == "__main__":
    np.random_seed = 42
    docs_path = "./processed_docs.pkl"
    assert os.path.exists(docs_path), "Processed docs could not be found in this\
        directory. They will be processed now"

    # docs is a dictionary with doc-ids as keys, value: lists of preprocessed words
    docs = get_processed_docs()

    # print example document
    # print(docs["AP891026-0263"])

    embeddings = train_skipgram(docs)
    




