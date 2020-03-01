from gensim_doc2vec import *
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from read_ap import process_text, get_processed_docs
from process_json import *

import os
from collections import Counter
from time import time
import read_ap
from tqdm import tqdm
import pytrec_eval
import json
import numpy as np


# initialize for gridsearch
vector_dimensions = [200, 300, 400, 500]
window_sizes = [5, 10, 15, 20]
min_counts = [250, 50, 15, 5, 2]


default_window_size = 5
default_vector_dim = 300
default_min_count = 50

# create directories
os.makedirs("json_files", exist_ok=True)
os.makedirs("models", exist_ok=True)


# load documents
processed_docs = get_processed_docs()
docs = processed_docs.values()
doc_keys = processed_docs.keys()
idx2key = {i: key for i, key in enumerate(doc_keys)}

# convert to TaggedDocuments so that gensim can work with them
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
print(f"{len(docs)} docs are loaded")


# vector dimensions
for vec_dim in vector_dimensions:
    print(f"\nvec_dim: {vec_dim}")
    model, model_name = training(documents, default_min_count,
                                 vector_dim=vec_dim, window_size=default_window_size)

    if not os.path.isfile(f"./json_files/benchmark_{model_name}.json"):
        _ = benchmark(model, model_name, docs, idx2key)

        # window sizes
for win_size in window_sizes:
    print(f"\nwindow size: {win_size}")
    model, model_name = training(documents, default_min_count,
                                 vector_dim=default_vector_dim, window_size=win_size)

    if not os.path.isfile(f"./json_files/benchmark_{model_name}.json"):
        _ = benchmark(model, model_name, docs, idx2key)


for min_c in min_counts:
    print(f"\nmax min count: {min_c}")
    model, model_name = training(documents, min_c,
                                 vector_dim=default_vector_dim, window_size=default_window_size)

    if not os.path.isfile(f"./json_files/benchmark_{model_name}.json"):
        _ = benchmark(model, model_name, docs, idx2key)


print("Below the processed results from all the json files")
process_json()
