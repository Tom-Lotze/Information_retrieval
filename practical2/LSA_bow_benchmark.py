from gensim.models import LsiModel
import download_ap
import read_ap
from tqdm import tqdm
import lsa_bow
import pytrec_eval
import json

import numpy as np

# ensure dataset is downloaded
download_ap.download_dataset()

# pre-process the text
docs_by_id = read_ap.get_processed_docs()
idx2key = {i: key for i, key in enumerate(docs_by_id.keys())}


# read in the qrels
qrels, queries = read_ap.read_qrels()

model = LsiModel.load(f"./models/lsi_bin_filtered")
print("loaded model succesfully")

# load model
search_engine = lsa_bow.Search(
    model=model, model_type="lsi_bin", num_topics=500)
# collect results
overall_ser = {}
for qid in tqdm(qrels):
    query_text = queries[qid]
    results = search_engine.query(query_text)

    overall_ser[qid] = dict(
        [(idx2key[idx], np.float64(score)) for idx, score in results])

# run evaluation with `qrels` as the ground truth relevance judgements
# here, we are measuring MAP and NDCG, but this can be changed to
# whatever you prefer
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
metrics = evaluator.evaluate(overall_ser)

# dump this to JSON
# *Not* Optional - This is submitted in the assignment!
with open(f"./json_files/LSI_bow.json", "w") as writer:
    json.dump(metrics, writer, indent=1)
