import numpy as np
import torch
from read_ap import *
import pickle as pkl
import os
import torch.nn as nn
import torch.optim as optim
from skipgram import *
from collections import Counter


def create_vocab(docs, threshold=120):
    cntr = Counter()

    for (doc_id, doc) in docs.items():
        for word in doc:
            cntr[word] += 1

    vocabulary = {}

    i = 0

    for element in cntr:
        if cntr[element] > threshold:
            vocabulary[element] = i
            i += 1

    return vocabulary, cntr


if __name__ == "__main__":
    np.random_seed = 42
    docs_path = "./processed_docs.pkl"
    assert os.path.exists(docs_path), "Processed docs could not be found in this\
        directory. They will be processed now"

    # docs is a dictionary with doc-ids as keys, value: lists of
    # preprocessed words
    docs = get_processed_docs()
    docs_test = docs
    # docs_test = dict()

    # for doc_id in list(docs.keys())[:1000]:
    # docs_test[doc_id] = docs[doc_id]

    # print example document
    # print(docs["AP891026-0263"])

    vocab, counter = create_vocab(docs_test)

    SKIP = Skipgram(docs_test, vocab, counter, "mean")

    embeddings = train_skipgram(SKIP, docs_test)
