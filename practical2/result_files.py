# -*- coding: utf-8 -*-
# python 3
# @Author: TomLotze
# @Date:   2020-02-29 12:21:17
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-02-29 16:10:31

import read_ap
import gensim_doc2vec
import lsa_lda



def ranking_to_trec_style(rank_fn, model_name, idx2key):
    """ the ranking must be a list of tuples"""

    qrels, queries = read_ap.read_qrels()

    # ensure ranking is sorted
    with open(f'results_{modelname}.txt', 'a') as f:
        for qid in qrels: 
            query_text = queries[qid]
            ranking = rank_fn(query_text)
            for rank, (doc_id, score) in enumerate(ranking[:1000]):
                doc_key = idx2key[doc_id]
                f.write(f'{qid} Qo {doc_key} {rank} {score} run1\n')


if __name__ == "__main__":

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()
    idx2key = {i: key for i, key in enumerate(docs_by_id.keys())}

    # choose from Word2Vec, Doc2Vec, 
    modelname = "Doc2Vec"



    rank_fn = 

    ranking_to_trec_style(rank_fn, model_name, idx2key)