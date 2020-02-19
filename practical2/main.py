import numpy as np
import torch
from read_ap import *
import pickle as pkl 
import os
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cosine as cos_similarity
from skipgram import *










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