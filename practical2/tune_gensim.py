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


vector_dimensions = [200, 300, 400, 500]
window_sizes = [5, 10, 15, 20]
vocab_sizes = [10, 25, 50, 100, 200] * 1000

default_window_size = 5
default_vector_dim = 300
default_vocab_size = 25000


# load documents
processed_docs = get_processed_docs()
docs = processed_docs.values()
doc_keys = processed_docs.keys()
idx2key = {i: key for i, key in enumerate(doc_keys)}

# convert to TaggedDocuments so that gensim can work with them
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
print(f"Docs are loaded. {len(docs)} in total\n")



# vector dimensions
for vec_dim in vector_dimensions::
    model, model_name = training(docs, max_vocab_size=default_vocab_size,
        vector_dim=vec_dim, window_size=default_window_size)

    # if there is a json file already, stop
    if os.path.isfile(f"Benchmark_{model_name}.json"):
        continue

    _ = benchmark(model, model_name, docs, idx2key)


# window sizes
for win_size in window_sizes::
    model, model_name = training(docs, max_vocab_size=default_vocab_size,
        vector_dim=default_vector_dim, window_size=win_size)

    # if there is a json file already, stop
    if os.path.isfile(f"Benchmark_{model_name}.json"):
        continue

    _ = benchmark(model, model_name, docs, idx2key)



for max_vocab in vocab_sizes::
    model, model_name = training(docs, max_vocab_size=max_vocab,
        vector_dim=default_vector_dim, window_size=default_window_size)

    # if there is a json file already, stop
    if os.path.isfile(f"Benchmark_{model_name}.json"):
        continue

    _ = benchmark(model, model_name, docs, idx2key)



process_json()