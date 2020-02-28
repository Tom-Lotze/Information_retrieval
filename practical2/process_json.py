import json
import os
import numpy as np
from scipy.stats import ttest_rel
from itertools import combinations

def process_json():
    curr_path = os.getcwd()

    json_files = [file for file in os.listdir(curr_path) if file.endswith(".json")]
    print(json_files)
    scores = dict()

    for file in json_files:
        print("\n"+file)
        with open(file) as f:
            scores_dict = json.load(f)
        scores[file] = scores_dict
        avg_map_all = np.mean([scores_dict[query_scores]["map"] for query_scores in 
            scores_dict])
        avg_map_76_100 = np.mean([scores_dict[query_scores]["map"] for query_scores in 
            scores_dict if int(query_scores) in range(76, 101)])
        avg_ndcg_all = np.mean([scores_dict[query_scores]["ndcg"] for query_scores in 
            scores_dict])
        avg_ndcg_76_100 = np.mean([scores_dict[query_scores]["ndcg"] for query_scores in 
            scores_dict if int(query_scores) in range(76, 101)])

        print(f"avg_map_all: {avg_map_all:.4f}")
        print(f"avg_map_76-100: {avg_map_76_100:.4f}")
        print(f"avg_ndcg_all: {avg_ndcg_all:.4f}")
        print(f"avg_map_76-100: {avg_ndcg_76_100:.4f}")



    # run relevance tests for all models
    all_combinations = list(combinations(scores.keys(), 2))
    for key1, key2 in all_combinations:
        first_scores = sum([[qr["map"], qr["ndcg"]] for qr in scores[key1].values()], [])
        second_scores = sum([[qr["map"], qr["ndcg"]] for qr in scores[key2].values()], [])
        print(f"\nT-test between {key1} and {key2}: {ttest_rel(first_scores, second_scores)}")

    # compute t-test for following models
    # word2vec-doc2vec
    # word2vec-LSI
    # word2vec-LDA
    # doc2vec-LSI
    # doc2vec-LDA
    # LSI-LDA

    return 0


