from gensim.models import LsiModel, LdaModel, TfidfModel
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


def train(n_topics = num_topics):

    docs = read_ap.get_processed_docs()
    docs = [d for i,d in docs.items()]

    dictionary = corpora.Dictionary(docs)

    # save the dictionary
    with open(os.path.join(folder_path_objects, 'dictionary'), 'wb') as f:
        pickle.dump(dictionary, f)

    # create binary and regular bow corpus
    corpus_bow     = [dictionary.doc2bow(d) for d in docs]
    corpus_binary  = [[(i,1) for i,_ in d] for d in corpus_bow]

    # create tf-idf corpus
    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    # save corpuses
    with open(os.path.join(folder_path_objects, 'corpus_binary'), 'wb') as f:
        pickle.dump(corpus_binary, f)

    with open(os.path.join(folder_path_objects, 'corpus_tfidf'), 'wb') as f:
        pickle.dump(corpus_tfidf, f)


    # create models
    print(f'{time.ctime()} Start training LSA (binary bow)')
    lsi_bin = LsiModel(
        corpus = corpus_binary, 
        id2word = dictionary,
        num_topics = n_topics
    )

    print(f'{time.ctime()} Start training LSA (tf-idf)')
    lsi_tfidf = LsiModel(
        corpus = corpus_tfidf, 
        id2word = dictionary,
        num_topics = n_topics
    )

    print(f'{time.ctime()} Start training LDA (tf-idf)')
    lda_tfidf = LdaModel(
        corpus = corpus_tfidf, 
        id2word = dictionary,
        num_topics = n_topics,
        dtype = np.float64
    )

    # save models to disk    
    os.makedirs(folder_path_models, exist_ok=True)
    filepath_out = lambda model: os.path.join('models', f'{model}_{t}')

    lsi_bin.save(filepath_out('lsi_bin'))
    lsi_tfidf.save(filepath_out('lsi_tfidf'))
    lda_tfidf.save(filepath_out('lda_tfidf'))


def create_index(model):
    assert model in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']

    if 'tfidf' in model:
        corpus = get_corpus('tfidf')
    else:
        corpus = get_corpus('binary')

    m = get_model(model)
    dictionary = get_dictionary()

    index = similarities.SparseMatrixSimilarity(m[corpus], num_features=len(dictionary.token2id))

    filepath_out = os.path.join(folder_path_objects, f'index_{model}')

    index.save(filepath_out)






# helper functions
def get_index(model_type, num_topics):
    assert model_type in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']

    # if 'tfidf' in model:
    #     corpus = get_corpus('tfidf')
    # else:
    #     corpus = get_corpus('binary')

    filepath_in = os.path.join(folder_path_objects, f'index_{model_type}_{num_topics}')
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

        #print(f"q is : {q}")

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
    create_index('lsi_tfidf')
    create_index('lda_tfidf')






