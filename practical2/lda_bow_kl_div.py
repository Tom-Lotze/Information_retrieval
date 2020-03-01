from gensim.models import LdaModel, LdaMulticore
from gensim import corpora
from gensim.matutils import sparse2full, kullback_leibler
from tqdm import tqdm
import json
import read_ap
import time
import os
import pickle as pkl
import numpy as np
import pytrec_eval
num_topics = 500

models_path = './models'
objects_path = './objects'
results_path = './results'

os.makedirs(models_path, exist_ok=True)
os.makedirs(objects_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)


def train(n_topics=num_topics):
    '''Train LDA model'''

    docs = read_ap.get_processed_docs()

    docs = [d for i, d in docs.items()]

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=50)

    # save the dictionary
    with open('./objects/dictionary_lda', 'wb') as f:
        pkl.dump(dictionary, f)

    # creating bow
    print('creating bow corpus')
    corpus_bow = [dictionary.doc2bow(d) for d in docs]
    # creating binary bow
    print('creating binary bow')
    corpus_binary = [[(i, 1) for i, _ in d] for d in corpus_bow]

    # with open(os.path.join(objects_path, 'corpus'), 'wb') as f:
    #     pickle.dump(corpus_tfidf, f)

    print(f'{time.ctime()} Start training LDA (BOW)')
    lda_bow = LdaMulticore(
        workers=5,
        corpus=corpus_binary,
        id2word=dictionary,
        chunksize=1000,
        num_topics=n_topics,
        dtype=np.float64
    )

    # save models to disk
    os.makedirs(models_path, exist_ok=True)

    lda_bow.save(os.path.join(models_path, f'lda_bow_multi'))


def create_index(n_topics=num_topics):

    lda_bow = LdaModel.load(os.path.join(models_path, f'lda_bow_multi'))
    print('Loaded model')
    docs = read_ap.get_processed_docs()
    docs = [d for i, d in docs.items()]

    with open('./objects/dictionary_lda', 'rb') as f:
        dictionary = pkl.load(f)

    # creating bow
    print('creating bow corpus')
    corpus_bow = [dictionary.doc2bow(d) for d in docs]
    # creating binary bow
    print('creating binary bow')
    corpus_binary = [[(i, 1) for i, _ in d] for d in corpus_bow]

    corpus_full = [sparse2full(t_doc, n_topics)
                   for t_doc in lda_bow[corpus_binary]]

    with open('./objects/lda_bow_full', 'wb') as f:
        pkl.dump(corpus_full, f)

    return corpus_full


def get_sims(model, query, corpus_full, dictionary, n_topics):
    ''' get ranking for single query'''

    # avoid division by 0
    eps = 1e-8

    # process query
    query_processed = read_ap.process_text(query)
    query_bow = dictionary.doc2bow(query_processed)
    q_lda = sparse2full(model[query_bow], n_topics)
    q_lda += eps

    sims = []

    # loop over all docs
    for i, doc in enumerate(corpus_full):
        doc += eps
        sim = -1 * kullback_leibler(q_lda, doc)
        sims.append(sim)

    sim_ordered = sorted(enumerate(sims), key=lambda item: -1 * item[1])

    return sim_ordered


def get_ranking(n_topics=num_topics):
    ''' get ranking for all queries'''

    # load queries
    qrels, queries = read_ap.read_qrels()

    # load model
    lda_bow = LdaModel.load(os.path.join(models_path, 'lda_bow_multi'))

    # load corpus of full vectors
    with open('./objects/lda_bow_full', 'rb') as f:
        corpus_full = pkl.load(f)

    # load dictionary
    with open('./objects/dictionary_lda', 'rb') as f:
        dictionary = pkl.load(f)

    # process docs
    processed_docs = read_ap.get_processed_docs()
    doc_keys = processed_docs.keys()
    idx2key = {i: key for i, key in enumerate(doc_keys)}

    overall_ser = {}

    # loop over queries
    for qid in tqdm(qrels):
        query_text = queries[qid]
        sims = get_sims(lda_bow, query_text, corpus_full, dictionary, n_topics)

        overall_ser[qid] = dict([(idx2key[idx], np.float64(score))
                                 for idx, score in sims])

    with open('./objects/overal_ser_lda', 'wb') as f:
        pkl.dump(overall_ser, f)


def get_json():
    '''load overal_serr from pickle and create json'''

    with open('./objects/overal_ser_lda', 'rb') as f:
        overal_serr = pkl.load(f)

    qrels, queries = read_ap.read_qrels()

    print('pytreccing')

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overal_serr)

    print('dumping json')
    with open(f'./json_files/lda_bow_kl.json', 'w') as f:
        json.dump(metrics, f, indent=1)


# train(500)
# create_index(500)
# get_ranking()
get_json()
