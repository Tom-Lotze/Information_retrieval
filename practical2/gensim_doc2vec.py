
import os
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from read_ap import process_text, get_processed_docs
from collections import Counter
from time import time
import read_ap
from tqdm import tqdm
import pytrec_eval
import json
import numpy as np


def training(docs, max_vocab_size, vector_dim=300, window_size=2, verbose=False):
    """
    This function takes the processed documents and trains a gensim Doc2Vec
    model on it. If there is a saved model, it loads the model and returns that
    """
    begin = time()
    model_name = f"gensim_{len(docs)}_{max_vocab_size}_{vector_dim}_{window_size}.model"
    if verbose: print(f"model name: {model_name}")
    
    model = Doc2Vec(vector_size=vector_dim, window=window_size, workers=4, epochs=2, seed=42, max_vocab_size=max_vocab_size)
    if verbose: print(f"Model initialized in {time()-begin:.2f} seconds\n")

    try: 
        model = Doc2Vec.load(os.path.join("models", model_name))
        print("Model loaded from memory")
    except:
        if verbose: print("No saved model found in this folder. Training now")
        model.build_vocab(docs)
        print(f"Vocabulary is built in {time()-begin:.2f} seconds")
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
        print(f"Model is trained in {time()-begin:.2f} seconds")
        model.save(os.path.join("models", model_name))
        print(f"Model is saved as {model_name} in the models folder\n")

    if verbose: print(f"length vocab: {len(model.wv.vocab)}")
    return model, model_name


def sanity_check(model, docs):
    """
    This function checks the working of the model by taking every document its
    text as a query and ensuring that the document itself (which is ofcourse
    relevant to the query) is ranked number 1
    """
    ranks = []
    second_ranks = []
    for doc_id in range(len(docs)):
        inferred_vector = model.infer_vector(docs[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])
    cntr = Counter(ranks)
    return cntr


def rank(model, docs, query_raw):
    query = process_text(query_raw)
    query_vector = model.infer_vector(query)

    ranking = model.docvecs.most_similar([query_vector], topn=len(model.docvecs))

    return ranking

def benchmark(model, model_name, docs, idx2key):
    qrels, queries = read_ap.read_qrels()

    overall_ser = {}

    # Adopted version from the TFIDF benchmark test
    print("Running GENSIM Benchmark")
    # collect results
    for qid in tqdm(qrels): 
        query_text = queries[qid]
        results = rank(model, docs, query_text) 
        #print(results)
        overall_ser[qid] = dict([(idx2key[idx], score) for idx, score in results])
        
    print(overall_ser)

    #print(overall_ser[100])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    json_filename = f"./json_files/benchmark_{model_name}.json"

    # dump to JSON
    with open(json_filename, "w") as writer:
       json.dump(metrics, writer, indent=1)

    return json_filename



if __name__ == "__main__":
    np.random.seed(42)

    # retrieve docs as a list
    processed_docs = get_processed_docs()
    docs = processed_docs.values()
    doc_keys = processed_docs.keys()
    idx2key = {i: key for i, key in enumerate(doc_keys)}

    # convert to TaggedDocuments so that gensim can work with them
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
    print(f"Docs are loaded. {len(docs)} in total\n")

    # train the model
    model, model_name = training(documents, max_vocab_size=3000000, vector_dim=300, window_size=2, verbose=True)

    # perform benchmark on the model and jump to json file
    json_filename = benchmark(model, model_name, documents, idx2key)

    # sanity check takes a LONG time on full sized doc collection (>2h)
    #print(f"Sanity check:\n{sanity_check(model, documents)}\n")
    
    # ranking on example query
    
    query_raw = "Bloomberg did not perform well during the Democratic election debate"
    ranking = rank(model, documents, query_raw)
    print(f"Ranking (top 10) for the query \"{query_raw}\":\n{ranking[:10]}\n")
    for i in range(10):
        print(" ".join(processed_docs[idx2key[ranking[i][0]]]) + "\n")






