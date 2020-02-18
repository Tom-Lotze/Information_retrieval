
# This file implements a skip gram model (word2vec) using Pytorch

import numpy as np
import torch
from read_ap import *
import pickle as pkl 
import os
import torch.nn as nn

docs_path = "./processed_docs.pkl"
assert os.path.exists(docs_path), "Processed docs could not be found in this\
    directory. They will be processed now"

# docs is a dictionary with doc-ids as keys, value: lists of preprocessed words
docs = get_processed_docs()

# print example document
# print(docs["AP891026-0263"])


class Skipgram(nn.Module):
    def __init__(self, docs, output_dim=300, window_size=5):
        """
        Initialization of the skip gram model
        Arguments:
            - docs: documents on which the embeddings are learned
            - output_dim: dimension of the embeddings that are learned
            - window_size: window size in the skip gram model

        """
        super(MLP, self).__init__()

        # set tunable parameters
        self.output_dim = output_dim
        self.window_size = 5
        
        self.docs = docs

        self.layers = 



    def forward():
        """Perform a forward pass"""

    def word_embedding(word):
        pass
        nn.Embedding





    def doc_embedding(doc):
        #aggregate word embeddings
        pass




if __name__ == "__main__":
    docs_path = "./processed_docs.pkl"
    assert os.path.exists(docs_path), "Processed docs could not be found in this\
        directory. They will be processed now"

    # docs is a dictionary with doc-ids as keys, value: lists of preprocessed words
    docs = get_processed_docs()

    skipgram_model = Skipgram(docs)




