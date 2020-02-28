import download_ap
import read_ap
import trec
from tqdm import tqdm
import lsa_lda

# ensure dataset is downloaded
download_ap.download_dataset()

# pre-process the text
docs_by_id = read_ap.get_processed_docs()

# read in the qrels
qrels, queries = read_ap.read_qrels()



for model in ['lsi_bin', 'lsi_tfidf', 'lda_tfidf']:
    print(f"Running TFIDF Benchmark for {model}")
    
    # load model
    search_engine = lsa_lda.Search(model=model)
    
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
    with open(f"{model}.json", "w") as writer:
        json.dump(metrics, writer, indent=1)