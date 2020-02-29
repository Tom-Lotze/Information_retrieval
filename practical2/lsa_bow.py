from gensim.models import LsiModel
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
    with open('./objects/dictionary_lsi_bin', 'wb') as f:
        pickle.dump(dictionary, f)

    # create binary and regular bow corpus
    corpus_bow = [dictionary.doc2bow(d) for d in docs]
    corpus_binary = [[(i, 1) for i, _ in d] for d in corpus_bow]

    # # save corpuses
    with open(os.path.join(folder_path_objects, 'corpus_binary_lsi'), 'wb') as f:
        pickle.dump(corpus_binary, f)

    # create models
    # print(f'{time.ctime()} Start training LSA (binary bow)')
    # lsi_bin = LsiModel(
    #     corpus=corpus_binary,
    #     id2word=dictionary,
    #     chunksize=1000,
    #     num_topics=n_topics
    # )

    # # save models to disk
    # os.makedirs(folder_path_models, exist_ok=True)

    # lsi_bin.save('./models/lsi_bin_filtered')


def create_index(model):

    with open('./objects/corpus_binary_lsi', 'rb') as f:
        corpus = pickle.load(f)

    m = LsiModel.load('./models/lsi_bin_filtered')

    with open('./objects/dictionary_lsi_bin', 'rb') as f:
        dictionary = pickle.load(f)

    index = similarities.SparseMatrixSimilarity(
        m[corpus], num_features=len(dictionary.token2id))

    index.save('./objects/index_lsi_bin_filtered')


# helper functions
def get_index(model_type, num_topics):
    assert model_type in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']

    # if 'tfidf' in model:
    #     corpus = get_corpus('tfidf')
    # else:
    #     corpus = get_corpus('binary')

    filepath_in = os.path.join(
        folder_path_objects, f'index_{model_type}_filtered')
    index = similarities.MatrixSimilarity.load(filepath_in)

    return index


def get_dictionary():
    with open(os.path.join(folder_path_objects, 'dictionary'), 'rb') as f:
        return pickle.load(f)


def get_corpus(corpus):
    assert corpus in ['binary', 'tfidf']

    with open(os.path.join(folder_path_objects, f'corpus_{corpus}'), 'rb') as f:
        return pickle.load(f)


def get_model(model, num_topics=500):
    assert model in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']

    filepath = os.path.join(folder_path_models, f'{model}_{num_topics}')

    if 'lsi' in model:
        return LsiModel.load(filepath)
    else:
        return LdaModel.load(filepath)


class Search:

    def __init__(self, model, model_type, num_topics):
        assert model_type in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']

        self.index = get_index(model_type, num_topics)
        self.dictionary = get_dictionary()
        self.model = model

    def query(self, q):

        # assert model in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']

        # get doc representation
        q = read_ap.process_text(q)
        q = self.dictionary.doc2bow(q)

        # print(f"q is : {q}")

        # convert vector to LSI space
        vec_query = self.model[q]

        sims = self.index[vec_query]

        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        # for i, s in enumerate(sims):
        #     print(s, documents[i])

        return sims


if __name__ == '__main__':
    t = int(time.time())

    # train models
    train(n_topics=500)

    # create indices
    create_index('lsi_bin')
    # create_index('lsi_tfidf')
    # create_index('lda_tfidf')
