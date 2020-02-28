# from gensim_doc2vec import *
# from gensim.test.utils import common_texts
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import read_ap
from process_json import *
from gensim.models import LsiModel, LdaModel, TfidfModel

import os
from collections import Counter
from time import time
import read_ap
from tqdm import tqdm
import pytrec_eval
import json
import numpy as np


# initialize for gridsearch
vector_n_topics = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

# create directories
os.makedirs("json_files", exist_ok=True)
os.makedirs("models", exist_ok=True)


# load documents
docs = read_ap.get_processed_docs()
docs = [d for i,d in docs.items()]
dictionary = corpora.Dictionary(docs)

# convert to TaggedDocuments so that gensim can work with them
corpus_bow = [dictionary.doc2bow(d) for d in docs]
tfidf = TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

print(f"{len(docs)} docs are loaded")


for num_topics in vector_n_topics:
    print(f"\nvec_dim: {vec_dim}")

    print(f'{time.ctime()} Start training LSA (tf-idf)')
    lsi_tfidf = LsiModel(
        corpus = corpus_tfidf, 
        id2word = dictionary,
        num_topics = num_topics
    )

    if not os.path.isfile(f"./json_files/benchmark_lsi_tfidf.json"):
        _ = benchmark(model, model_name, docs, idx2key)

    print(f'{time.ctime()} Start training LDA (tf-idf)')
    lda_tfidf = LdaModel(
        corpus = corpus_tfidf, 
        id2word = dictionary,
        num_topics = num_topics,
        dtype = np.float64
    )

    if not os.path.isfile(f"./json_files/benchmark_lda_tfidf.json"):
        _ = benchmark(model, model_name, docs, idx2key)

        # window sizes



print("Below the processed results from all the json files")
process_json()
