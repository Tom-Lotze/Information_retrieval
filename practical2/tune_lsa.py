# from gensim_doc2vec import *
# from gensim.test.utils import common_texts
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import read_ap
from gensim.models import LsiModel, TfidfModel
import os
import time
import read_ap
import json
from process_json import *

# import lsa_lda_benchmark
from gensim import corpora, similarities

# initialize for gridsearch
vector_n_topics = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

# create directories
folder_path_models = 'models'
folder_path_objects = 'objects'
folder_path_results = 'results'

os.makedirs("json_files", exist_ok=True)
os.makedirs("models", exist_ok=True)


# load documents
docs = read_ap.get_processed_docs()
docs = [d for i, d in docs.items()]
dictionary = corpora.Dictionary(docs)

# convert to TaggedDocuments so that gensim can work with them
corpus_bow = [dictionary.doc2bow(d) for d in docs]
tfidf = TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

print(f"{len(docs)} docs are loaded")


for num_topics in vector_n_topics:

    print(f'{time.ctime()} Start training LSA (tf-idf) num_topics = {num_topics}')
    lsi_tfidf = LsiModel(
        corpus=corpus_tfidf,
        id2word=dictionary,
        num_topics=num_topics
    )

    # save model
    fp_model_out = os.path.join(folder_path_models, f'lsi_tfidf_{num_topics}')
    lsi_tfidf.save(fp_model_out)

    index = similarities.SparseMatrixSimilarity(
        lsi_tfidf[corpus_tfidf],
        num_features=len(dictionary.token2id)
    )

    # save index
    fp_index_out = os.path.join(
        folder_path_objects, f'index_lsi_tfidf_{num_topics}')
    index.save(fp_index_out)


print("Below the processed results from all the json files")
process_json()
