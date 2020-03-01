# IR1-practical
In this repo all the practical assignments for Information Retrieval 1 can be found. The course was taken in Spring 2020.

### Collaborators
- Berend Jansen
- Tom Lotze
- Ruth Wijma

## General instructions
In this assignment, we created various models for information retrieval: Word2Vec, Doc2Vec, LSA/LSI (using TF-IDF and BOW) and LDA
Below, an overview is given of all files in this repo, and their function. The performance of the models is measured based on Mean Average Precision (MAP) and normalized Cumulative Distributive Gain (nDCGD) on rankings for a set of (provided) queries.

## Files
### Models
- tf_idf.py
- skipgram.py - word2vec mode
- gensim_doc2vec.py
- lsa_bow.py - LSA for binary BOW
- lsa_tfidf.py - LSA for TFIDF
- lda_bow_kl_div.py - LDA model with BOW and KL divergence ranking and pytrec eval

### Tuning and benchmarking
- tune_gensim.py: Gridsearch on the vocabulary size, embedding dimension and window size, saves all the models
- tune_lsa.py: Search for optimal number of topics for the LSA TFIDF model
- lsa_tfidf_benchmark.py: benchmark test for various number of topic LSA TF-IDF
- lsa_bow_benchmark: benchmark test for the one LSA BoW model

### Analysis
- process_json.py: Perform analysis on the json files (average MAP and nDCG for test and all queries per model) and significance testing (t-test) between the different models.
- query_analysis.py: finding the variance in query performance and best and worst performing queries per model.
- result_files.py: creates the TREC style result files
- lsa_lda_analysis.ipynb: extracting the topics from the LSI model

## Plotting
- plot_gensim.py: plot MAP of gridsearch models for Doc2Vec
- plot_LSI.py: plot the MAP results of various number of topics for LSI

## Subfolders
- datasets 
- objects - used for storing dictionaries/corpora/etc
- models - used for storing trained models
- json_files: JSON files with the 
- results: TREC style result files

## Instructions word2vec skipgram.py
   - run skipgram.py to train the model. The model will only train on 10% percent of the documents due to limited resources.
  model. Training will take a while. After training, the model will aggregate all documents to embeddings that can be used for retrieval tasks. Use model.rank_docs(query) to get a ranking per query.

## Instructions gensim_doc2vec.py
   - run gensim_doc2vec.py to create a json file with the rankings for each query.
   - individual queries can be used by running rank(model, docs, query_raw)

## Instructions lsa_bow.py
 - run lsa_bow.py and it will train and create an index. This index can be used for retrieval. Use search.query(q) where q is the query to retrieve a ranking of documents.

## Instructions lsa_tfidf.py
 - Similar to lsa_bow.py

## Instructions lda_bow_kl_div.py
   - run lda_bow_kl_div.py to train the model, create the index, get the ranking for all queries from the dataset and dump the json file.
   - for individual queries, use the individual_query(query_text) function

  

