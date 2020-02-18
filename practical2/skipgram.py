
# This file implements a skip gram model (word2vec) using Pytorch

import numpy as np
import torch
from read_ap import *
import pickle as pkl 
import os

docs_path = "./processed_docs.pkl"
assert os.path.exists(docs_path), "Processed docs could not be found in this\
    directory. They will be processed now"

# docs is a dictionary with doc-ids as keys, value: lists of preprocessed words
docs = get_processed_docs()

# print example document
# print(docs["AP891026-0263"])

# set tunable parameters
output_dim = 300
window_size = 5




