# This file implements a skip gram model (word2vec) using Pytorch
import os
import numpy as np
import torch
from read_ap import *
import pickle as pkl
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
import pytrec_eval
import json
from tqdm import tqdm


class Skipgram(nn.Module):
    def __init__(self, docs, vocab, counter, aggregation_function,
                 embedding_dim=300, window_size=5):
        """
        Initialization of the skip gram model
        Arguments:
            - docs: documents on which the embeddings are learned
            - output_dim: dimension of the embeddings that are learned
            - window_size: window size in the skip gram model

        """
        super(Skipgram, self).__init__()

        # set tunable parameters
        self.embedding_dim = embedding_dim
        self.window_size = 5
        self.nr_epochs = 2000
        self.k = 5

        self.counter = counter

        # vocab needs to be made and filtered on infrequent words
        self.docs = docs
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)

        self.aggr_function = aggregation_function

        self.target_embedding = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True)
        self.context_embedding = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True)

        nn.init.uniform_(self.target_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.context_embedding.weight, 0, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_u, pos_v, neg_v):
        """Perform a forward pass on the tuple of two one hot encodings
        Input:
        - pos_u : list of positive target word indices
        - pos_v : list of positive context word indices
        - neg_v : list of negative context word indices

        """

        # indices for positive and negative examples
        pos_u = Variable(torch.Tensor(pos_u).long())
        pos_v = Variable(torch.Tensor(pos_v).long())
        neg_v = Variable(torch.Tensor(neg_v).long())

        # define batch size
        batch_size = len(pos_u) + len(pos_u) * self.k

        # target embedding (index the embeddings)
        pos_t_E = self.target_embedding(pos_u)

        # positive context embeddings
        pos_E = self.context_embedding(pos_v)

        # positie scores
        pos_score = torch.sum(torch.mul(pos_t_E, pos_E).squeeze(), dim=1)
        pos_score = torch.log(self.sigmoid(pos_score))

        # negative context embeddings
        neg_E = self.context_embedding(neg_v)

        print(f'neg_e: {neg_E.shape}, pos_E: {pos_t_E.shape}')
        # negative scores
        neg_score = torch.bmm(neg_E, pos_t_E.unsqueeze(2))
        neg_score = torch.sum(neg_score, dim=1)

        neg_score = torch.log(self.sigmoid(-neg_score)).squeeze()

        # return loss
        return -(pos_score.sum() + neg_score.sum()) / batch_size

    def word_embedding(self, word):
        ''' Take word as string and return embedding'''
        word_index = self.word_to_idx(word, self.vocab)
        return self.target_embedding(word_index)

    def word_to_idx(self, word):
        """Returns the index of the word (string) in the vocabulary (dict)"""
        return torch.Tensor([self.vocab[word]]).long()

    def idx_to_word(self, idx, inv_vocab):
        """
        Returns the word (string) corresponding to the index in the vocabulary
        (dict)
        """
        return inv_vocab[idx]

    def get_neg_sample_pdf(self, counter):
        ''' generate probability distribution for negative sampling'''
        denominator = np.sum([np.power(counter[w], 3/4)
                              for w in self.vocab.keys()])

        sampling_list = [np.power(counter[word], 3/4)
                         for word in self.vocab.keys()] / denominator

        return sampling_list

    def neg_sample(self, counter, pdf, k):
        ''' sample k negative training examples'''
        return np.random.choice(list(self.vocab.keys()), p=pdf, size=k)

    def most_similar(self, word):
        """Find most similar word given a target word"""
        word_idx = self.word_to_idx(process_text(word)[0])
        cossim = nn.CosineSimilarity()

        # create counter dict
        scores = Counter()

        # loop over all words in vocab
        for w, i in self.vocab.items():
            if i != word_idx:
                sim = cossim(self.target_embedding(
                    word_idx), self.target_embedding(torch.Tensor([i]).long()))
                # store cossims
                scores[w] = sim.item()

        # return most common
        return scores.most_common(10)

    def aggregate_doc(self, doc):
        '''Get doc_id and create vector by aggregating'''
        with torch.no_grad():
            # get the words that are in vocab
            rel_words = [w for w in doc if w in self.vocab.keys()]
            # get embeddings for words
            embeddings_tensor = torch.empty(
                len(rel_words), self.embedding_dim)

            # get embeddings for all words
            for i, word in enumerate(rel_words):
                if word in self.vocab.keys():
                    word_idx = self.word_to_idx(word)
                    word_E = self.target_embedding(word_idx)
                    embeddings_tensor[i, :] = word_E
                    del word_E
            # mean over word embeddings
            doc_E = torch.mean(embeddings_tensor, dim=0)
        return doc_E

    def aggregate_all_docs(self):
        ''' aggregate all docs to save time during retrieval'''
        agg_docs = torch.empty(len(self.docs.keys()), self.embedding_dim)

        # loop over all docs and get embedding and store
        for i, (doc_id, doc) in enumerate(self.docs.items()):
            print(f'Doc: {i}')
            doc_E = self.aggregate_doc(doc)
            agg_docs[i, :] = doc_E
            del doc_E

        # dump all aggregated docs to load them later
        with open('aggregated_docs.pt', 'wb') as f:
            pkl.dump(agg_docs, f)

        print('Aggregated all docs')

    def rank_docs(self, query):
        '''Ranks docs given query'''
        # get query and process
        q = process_text(query)

        # get query embedding
        q_E = self.aggregate_doc(q).view(1, self.embedding_dim)

        # initialize cosine similarity
        cossim = nn.CosineSimilarity(dim=1)

        # get doc ids
        doc_ids = [doc_id for doc_id in self.docs.keys()]

        # initialize dictionary as counter object
        scores = Counter()

        # load all doc embeddings
        with open('./aggregated_docs.pt', 'rb') as f:
            agg_docs = pkl.load(f)

        # batch cosine similarity over docs and query
        sim = cossim(q_E, agg_docs)

        # sort the indices
        sort_indx = torch.argsort(sim, descending=True)

        # return list of tuples
        scores = [(doc_ids[idx], sim[idx]) for idx in sort_indx]

        return scores


def get_batches(model, docs, batch_size, pdf):
    ''' generate batches '''

    # window size left and right
    top = int(model.window_size/2)

    pos_batch = []
    neg_batch = []

    # shuffle the docs for this epoch
    docs_list = np.array(list(docs.values()))
    np.random.shuffle(docs_list)

    for j, doc in enumerate(docs_list):
        # pad the doc for edge cases
        padded_doc = ["NULL"] * top + doc + ["NULL"] * top

        print(f'Doc: {j}')

        # loop over words in doc
        for i, target_word in enumerate(doc):

            # if target word not in vocab we skip it
            if target_word not in model.vocab.keys():
                continue

            i += top

            # define the window
            window = padded_doc[i-top: i]+padded_doc[i+1: i+top+1]

            # positive training pairs
            pos_pairs = [(model.word_to_idx(target_word), model.word_to_idx(
                c)) for c in window if c != "NULL" and c in model.vocab.keys()]

            # add to batch
            pos_batch += pos_pairs

            # if larger than batch size, sample negative
            if len(pos_batch) > batch_size:  # FIX
                for pos_x in pos_batch:
                    neg_batch.append([model.word_to_idx(c).item()
                                      for c in model.neg_sample(model.counter,
                                                                pdf, model.k)])
                yield (pos_batch, neg_batch)
                pos_pairs = []
                neg_batch = []
                pos_batch = []


def train_skipgram(model, docs):
    # numpy seed
    # set torch seed
    torch.manual_seed(42)
    np.random.seed(42)

    # set optimizer HYPERPARAMS?
    optimizer = optim.SparseAdam(model.parameters())

    # batch size
    batch_size = 256

    # get pdf for negative sampling
    pdf = model.get_neg_sample_pdf(model.counter)

    # epoch
    for epoch in range(model.nr_epochs):
        print(f"Epoch nr {epoch}")

        # get batch of positive and negative examples
        for step, (pos_batch, neg_batch) in enumerate(get_batches(model, docs,
                                                                  batch_size,
                                                                  pdf)):
            # print(pos_batch, neg_batch)
            optimizer.zero_grad()

            # extracht words
            pos_u = [x[0].item() for x in pos_batch]
            pos_v = [x[1].item() for x in pos_batch]
            neg_v = neg_batch

            # forward pass
            loss = model.forward(pos_u, pos_v, neg_v)

            if step % 50 == 0:
                print(f'at step {step}: loss: {loss.item()}')

            # backprop
            loss.backward()
            optimizer.step()

        # save model
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(model.state_dict(),
                   f'./models/trained_w2v_epoch_{epoch}.pt')


def benchmark(model):
    qrels, queries = read_qrels()

    overall_ser = {}

    # Adopted version from the TFIDF benchmark test
    print("Running GENSIM Benchmark")
    # collect results
    for qid in tqdm(qrels):
        query = queries[qid]
        results = model.rank_docs(query)
        # print(results)
        overall_ser[qid] = dict([(idx, score.item())
                                 for idx, score in results])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # dump to JSON
    with open("word2vec.json", "w") as writer:
        json.dump(metrics, writer, indent=1)
