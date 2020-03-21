import json
import os
import numpy as np
from scipy.stats import ttest_rel
from itertools import combinations


def process_json():
    curr_path = os.path.join(os.getcwd(), "listwise_berend/json_files/")

    json_files = [file for file in os.listdir(
        curr_path) if file.endswith("ERR.json")]
    # print(json_files)
    ndcg_on_test = dict()

    for file in json_files:
        if not "TEST" in file:
            continue
        print("\n"+file)
        with open(os.path.join(curr_path, file)) as f:
            scores_dict = json.load(f)
        ndcg_score = scores_dict["ndcg"][0]
        print(f"nDCG score: {ndcg_score}")
        ndcg_on_test[file] = ndcg_score

    max_ndcg_40_key = max(ndcg_on_test.keys(),
                          key=lambda key: ndcg_on_test[key])
    max_value = ndcg_on_test[max_ndcg_40_key]
    average_ndcg_40 = np.mean(list(ndcg_on_test.values()))
    var_ndcg_40 = np.var(list(ndcg_on_test.values()))

    return max_ndcg_40_key, max_value, average_ndcg_40, var_ndcg_40


if __name__ == "__main__":
    best_setting, best_ndcg, average, variance = process_json()
    print(
        f"\nbest config: {best_setting}\nNDCG score on test: {best_ndcg}\naverage NDCG on test: {average}\nNDCG variance: {variance}")
