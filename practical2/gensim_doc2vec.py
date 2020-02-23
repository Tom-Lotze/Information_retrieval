

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from read_ap import process_text, get_processed_docs
from collections import Counter
from time import time


def training(docs):

    begin = time()
    
    model_name = f"./gensim_{len(docs)}.model"
    
    model = Doc2Vec(vector_size=300, window=2, min_count=50, workers=4, epochs=10, seed=42)

    print(f"Model initialized in {time()-begin:.2f} seconds\n")

    try: 
        model = Doc2Vec.load(model_name)
        print("Model loaded from memory")
    except:
        print("No saved model found in this folder. Training now")

        model.build_vocab(documents)

        print(f"Vocabulary is built in {time()-begin:.2f} seconds\n")

        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        print(f"Model is trained in {time()-begin:.2f} seconds\n")

        model.save(model_name)

        print(f"Model is saved as {model_name}")

    return model


def sanity_check(model, docs):
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


def rank(model, docs, query):
    query_vector = model.infer_vector(query)

    ranking = model.docvecs.most_similar([query_vector])

    return ranking



if __name__ == "__main__":

    # retrieve docs as a list
    docs = list(get_processed_docs().values())
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

    print(f"Docs are loaded. {len(docs)} in total\n")

    model = training(documents)

    query_raw = "Bloomberg did not perform well during the Democratic election debate"

    query = process_text(query_raw)

    # sanity check takes a LONG time on full sized doc collection (>1h)
    #print(f"Sanity check:\n{sanity_check(model, documents)}\n")

    ranking = rank(model, documents, query)

    #print(f"{ranking}\n")

    for i in range(10):
        print(docs[ranking[i][0]])






