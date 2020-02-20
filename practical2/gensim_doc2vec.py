

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from read_ap import process_text
from collections import Counter


def training(docs):

    print("Docs are loaded\n")

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(vector_size=300, window=2, min_count=50, workers=4, epochs=10, seed=42)

    print("Model initialized\n")

    model.build_vocab(documents)

    print("Vocabulary is built\n")

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print("Model is trained\n")

    vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
    print(f"Inferred vector for testing: {vector}")

    return model


def sanity_check(model, docs):
    ranks = []
    second_ranks = []
    for doc_id in range(len(docs)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])

    return Counter(ranks)


def rank(model, docs, query):
    query_vector = model.infer_vector(query)

    ranking = model.docvecs.most_similar([inferred_vector])

    return ranking



if __name__ == "__main__":

    # retrieve docs as a list
    docs = list(get_processed_docs().values())

    model = training()

    query_raw = "Bloomberg did not perform well during the Democratic election debate"

    query = process_text(query_raw)

    print(sanity_check(model, docs))

    ranking = rank(model, docs, query)

    print(ranking)






