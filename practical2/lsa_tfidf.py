from gensim.models import TfidfModel, LsiModel
from gensim import corpora, similarities
import read_ap
import time
import os
import pickle
import numpy as np

# constants
folder_path_models = 'models'
folder_path_objects = 'objects'
folder_path_results = 'results'
num_topics = 500

os.makedirs(folder_path_models, exist_ok=True)
os.makedirs(folder_path_objects, exist_ok=True)
os.makedirs(folder_path_results, exist_ok=True)


def train(n_topics=num_topics):

    docs = read_ap.get_processed_docs()
    docs = [d for i, d in docs.items()]

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=50)

    # save the dictionary
    with open(os.path.join(folder_path_objects,
                           'dictionary_lsi_bow'), 'wb') as f:
        pickle.dump(dictionary, f)

    # create binary and regular bow corpus
    corpus_bow = [dictionary.doc2bow(d) for d in docs]

    # create tf-idf corpus
    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    with open(os.path.join(folder_path_objects,
                           'corpus_lsi_tfidf'), 'wb') as f:
        pickle.dump(corpus_tfidf, f)

    # create models
    print(f'{time.ctime()} Start training LSI (tf-idf)')
    lsi_tfidf = LsiModel(
        corpus=corpus_tfidf,
        id2word=dictionary,
        num_topics=n_topics
    )

    # save models to disk
    os.makedirs(folder_path_models, exist_ok=True)
    def filepath_out(model): return os.path.join('models', f'{model}_{t}')

    lsi_tfidf.save(filepath_out('lsi_tfidf'))


def create_index(model):
    corpus = get_corpus('tfidf')

    m = get_model(model)
    dictionary = get_dictionary()

    index = similarities.SparseMatrixSimilarity(
        m[corpus], num_features=len(dictionary.token2id))

    filepath_out = os.path.join(folder_path_objects, f'index_{model}')

    index.save(filepath_out)


# helper functions
def get_index(model_type, num_topics):

    # corpus = get_corpus('tfidf')

    filepath_in = os.path.join(
        folder_path_objects, f'index_{model_type}_{num_topics}')
    index = similarities.MatrixSimilarity.load(filepath_in)

    return index


def get_dictionary():
    with open(os.path.join(folder_path_objects,
                           'dictionary_lsi_bow'), 'rb') as f:
        return pickle.load(f)


def get_corpus(corpus):
    with open(os.path.join(folder_path_objects,
                           f'corpus_lsi_tfidf'), 'rb') as f:
        return pickle.load(f)


def get_model(model, num_topics=500):
    assert model == 'tfidf'
    filepath = os.path.join(folder_path_models, f'{model}_{num_topics}')
    return LsiModel.load(filepath)


class Search:

    def __init__(self, model, model_type, num_topics):
        assert model_type in ['lsi_tfidf']

        self.index = get_index(model_type, num_topics)
        self.dictionary = get_dictionary()
        self.model = model

    def query(self, q):

        # get doc representation
        q = read_ap.process_text(q)
        q = self.dictionary.doc2bow(q)

        # print(f"q is : {q}")

        # convert vector to LSI space
        vec_query = self.model[q]

        sims = self.index[vec_query]

        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        return sims


if __name__ == '__main__':
    t = int(time.time())

    # train models
    train(n_topics=500)

    # create indices
    create_index('lsi_tfidf')
