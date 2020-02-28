import download_ap
import read_ap
import trec
from tqdm import tqdm
import lsa_lda

vector_n_topics = [10, 50, 100, 500, 1000, 2000, 5000, 10000]


# ensure dataset is downloaded
download_ap.download_dataset()

# pre-process the text
docs_by_id = read_ap.get_processed_docs()

# read in the qrels
qrels, queries = read_ap.read_qrels()

for num_topics in vector_n_topics:


    print(f"Running TFIDF Benchmark for {model}")
    lsi_tfidf = LsiModel(
        corpus = corpus_tfidf, 
        id2word = dictionary,
        num_topics = num_topics
    )
    
    # load model
    search_engine = lsa_lda.Search(model='lsi_tfidf', num_topics=num_topics)
    
    # collect results
    overall_ser = {}
    for qid in tqdm(qrels): 
        query_text = queries[qid]
        results = search_engine.query(query_text)

        overall_ser[qid] = dict(results)



    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to 
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open(f"./json_files/{model}_{num_topics}.json", "w") as writer:
        json.dump(metrics, writer, indent=1)


