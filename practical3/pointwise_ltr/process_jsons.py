import json
import os
import numpy as np
from scipy.stats import ttest_rel
from itertools import combinations

def process_json():




    curr_path = os.path.join(os.getcwd(), "pointwise_ltr/json_files/")

    json_files = [file for file in os.listdir(curr_path) if file.endswith(".json")]
    #print(json_files)
    ndcg_at_40 = dict()

    for file in json_files:
        #print("\n"+file)
        with open(os.path.join(curr_path, file)) as f:
            scores_dict = json.load(f)
        ndcg_at_40[file] = scores_dict["39"]["ndcg"][0]


        # avg_map_all = np.mean([scores_dict[query_scores]["map"] for query_scores in
        #     scores_dict])
        # avg_map_76_100 = np.mean([scores_dict[query_scores]["map"] for query_scores in
        #     scores_dict if int(query_scores) in range(76, 101)])
        # avg_ndcg_all = np.mean([scores_dict[query_scores]["ndcg"] for query_scores in
        #     scores_dict])
        # avg_ndcg_76_100 = np.mean([scores_dict[query_scores]["ndcg"] for query_scores in
        #     scores_dict if int(query_scores) in range(76, 101)])

        # print(f"avg_map_all: {avg_map_all:.4f}")
        # print(f"avg_map_76-100: {avg_map_76_100:.4f}")
        # print(f"avg_ndcg_all: {avg_ndcg_all:.4f}")
        # print(f"avg_ndcg_76-100: {avg_ndcg_76_100:.4f}")

    #print(ndcg_at_40)

    max_ndcg_40_key = max(ndcg_at_40.keys(), key=lambda key: ndcg_at_40[key])
    max_value = ndcg_at_40[max_ndcg_40_key]
    average_ndcg_40 = np.mean(list(ndcg_at_40.values()))
    var_ndcg_40 = np.var(list(ndcg_at_40.values()))

    return max_ndcg_40_key, max_value, average_ndcg_40, var_ndcg_40

if __name__ == "__main__":
    best_setting, best_ndcg, average, variance = process_json()
    print(f"best config: {best_setting}\nNDCG score at 40 epochs: {best_ndcg}\naverage NDCG at 40: {average}\nNDCG variance: {variance}")


