# -*- coding: utf-8 -*-
# python 3
# @Author: TomLotze
# @Date:   2020-02-29 12:21:17
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-02-29 23:12:53

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import LsiModel

import read_ap
import gensim_doc2vec
import lsa_lda
import skipgram

from tqdm import tqdm
import torch
import pickle
import lsa_bow
import os






######## DOC2VEC #############

# def ranking_to_trec_style(rank_fn, model_name, model, docs, idx2key):
#     """ the ranking must be a list of tuples"""

#     qrels, queries = read_ap.read_qrels()

#     # ensure ranking is sorted
#     with open(f'results_{model_name}.txt', 'a') as f:
#         for qid in tqdm(qrels): 
#             query_text = queries[qid]
#             ranking = rank_fn(model, docs, query_text)
#             for rank, (doc_id, score) in enumerate(ranking[:1000]):
#                 doc_key = idx2key[doc_id]
#                 f.write(f'{qid} Q0 {doc_key} {rank} {score} run1\n')


# if __name__ == "__main__":
#     os.makedirs("results", exist_ok=True)

#     # pre-process the text
#     docs_by_id = read_ap.get_processed_docs()
#     docs = docs_by_id.values()
#     idx2key = {i: key for i, key in enumerate(docs_by_id.keys())}

#     # choose from Word2Vec, Doc2Vec, LSI_bow, LSI_tfidf, 
#     model_name = "Doc2Vec"
#     model = Doc2Vec.load("./models/gensim_164557.model")


#     rank_fn = gensim_doc2vec.rank

#     ranking_to_trec_style(rank_fn, model_name, model, docs, idx2key)




########   WORD2VEC    ########
# if __name__ == "__main__":
    # os.makedirs("results", exist_ok=True)

#     # pre-process the text
#     docs_by_id = read_ap.get_processed_docs()
#     docs = docs_by_id.values()
#     idx2key = {i: key for i, key in enumerate(docs_by_id.keys())}

#     # choose from Word2Vec, Doc2Vec, LSI_bow, LSI_tfidf, 
#     model_name = "Word2Vec"
#     with open("counter_word2vec.pt", "rb") as counter:
#         with open("vocab_word2vec.pt", "rb") as vocab:
#             model = skipgram.Skipgram(docs_by_id, pickle.load(vocab), pickle.load(counter), "mean")
#     model.load_state_dict(torch.load("./models/trained_w2v_epoch_1.pt"))

#     qrels, queries = read_ap.read_qrels()

#     # ensure ranking is sorted
#     with open(f'results_{model_name}.txt', 'a') as f:
#         for qid in tqdm(qrels): 
#             query_text = queries[qid]
#             ranking = model.rank_docs(query_text)
#             for rank, (doc_id, score) in enumerate(ranking[:1000]):
#                 f.write(f'{qid} Q0 {doc_id} {rank} {score} run1\n')



#####   LSI- TFIDF    ###### 
# if __name__ == "__main__":
    # os.makedirs("results", exist_ok=True)

#     # pre-process the text
#     docs_by_id = read_ap.get_processed_docs()
#     docs = docs_by_id.values()
#     idx2key = {i: key for i, key in enumerate(docs_by_id.keys())}

#     model = LsiModel.load(f"./models/lsi_tfidf_500")
#     model_name = "lsi_tfidf_500"
#     search_engine = lsa_lda.Search(model=model, model_type="lsi_tfidf", num_topics=500)

#     qrels, queries = read_ap.read_qrels()

#     # ensure ranking is sorted
#     with open(f'results_{model_name}.txt', 'a') as f:
#         for qid in tqdm(qrels): 
#             query_text = queries[qid]
#             ranking = search_engine.query(query_text)
#             for rank, (doc_id, score) in enumerate(ranking[:1000]):
#                 doc_key = idx2key[doc_id]
#                 f.write(f'{qid} Q0 {doc_key} {rank} {score} run1\n')


#####   LSI-BOW    ###### 
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()
    docs = docs_by_id.values()
    idx2key = {i: key for i, key in enumerate(docs_by_id.keys())}

    model = LsiModel.load(f"./models/lsi_bin_filtered")
    print("model loaded succesfully")
    model_name = "LSI_bow"

    search_engine = lsa_bow.Search(model=model, model_type="lsi_bin", num_topics=500)

    qrels, queries = read_ap.read_qrels()

    # ensure ranking is sorted
    with open(f'results/results_{model_name}.txt', 'a') as f:
        for qid in tqdm(qrels): 
            query_text = queries[qid]
            ranking = search_engine.query(query_text)
            for rank, (doc_id, score) in enumerate(ranking[:1000]):
                doc_key = idx2key[doc_id]
                f.write(f'{qid} Q0 {doc_key} {rank} {score} run1\n')














