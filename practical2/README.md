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
- skipgram.py (called from main.py)
- gensim_doc2vec.py
- lsa_bow.py
- lsa_lda.py

### Tuning and benchmarking
- tune_gensim.py: Gridsearch on the vocabulary size, embedding dimension and window size, saves all the models
- tune_lsa.py: Gridsearch on number of topics for the LSA model
- lsa_lda_benchmark.py: 
- lsa_benchmark.py: 

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
- objects
- models
- json_files: JSON files with the 
- results: TREC style result files

  

