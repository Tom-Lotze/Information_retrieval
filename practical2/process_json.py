import json
import os
import numpy as np


curr_path = os.getcwd()

for file in os.listdir(curr_path):
    if file.endswith(".json"):
        print(file)
        with open(file) as f:
            scores_dict = json.load(f)
        avg_map_all = np.mean([scores_dict[query_scores]["map"] for query_scores in 
            scores_dict])
        avg_map_76_100 = np.mean([scores_dict[query_scores]["map"] for query_scores in 
            scores_dict if int(query_scores) in range(76, 101)])
        avg_ndcg_all = np.mean([scores_dict[query_scores]["ndcg"] for query_scores in 
            scores_dict])
        avg_ndcg_76_100 = np.mean([scores_dict[query_scores]["ndcg"] for query_scores in 
            scores_dict if int(query_scores) in range(76, 101)])

        print(f"avg_map_all: {avg_map_all:.5f}")
        print(f"avg_map_76-100: {avg_map_76_100:.5f}")
        print(f"avg_ndcg_all: {avg_ndcg_all:.5f}")
        print(f"avg_map_76-100: {avg_ndcg_76_100:.5f}")



