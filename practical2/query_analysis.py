# -*- coding: utf-8 -*-
# python 3
# @Author: TomLotze
# @Date:   2020-02-29 22:35:11
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-03-01 10:30:43

import os
import json
import numpy as np
from collections import defaultdict

import read_ap


curr_path = os.path.join(os.getcwd(), "json_files/")

json_files = [file for file in os.listdir(curr_path) if file.endswith(".json")]
nr_files = len(json_files)
MAPS = {}
best_worse = {}

for file in json_files:
    with open(os.path.join("json_files", file)) as f:
        scores_dict = json.load(f)
    MAPS[file] = scores_dict
    best_worse[file] = {}
    # define best and worse as tuple of ID and average MAP
    max_key = max(scores_dict, key=lambda key: scores_dict[key]["map"])
    min_key = min(scores_dict, key=lambda key: scores_dict[key]["map"])
    best_worse[file]["best"] = (max_key, scores_dict[max_key]["map"])
    best_worse[file]["worst"] = (min_key, scores_dict[min_key]["map"])

query_ids = scores_dict.keys()
variances = {qid:np.var(np.array([MAPS[file][qid]['map'] for file in json_files])) for qid in query_ids}
sorted_variances = sorted(variances.items(), key=lambda kv: -kv[1])

qrels, queries = read_ap.read_qrels()


print("Queries with the highest variance between the retrieval models:")
for i in range(5):
    qid = sorted_variances[i][0]
    print(f"{i+1}: query {int(qid)} with variance in MAP of {sorted_variances[i][1]:.5f}")
    print(f"text: {queries[qid]}")

print(f"\nLowest and highest scoring queries per model:")
for file in json_files:
    print(f"{file[:-5]}")
    best = best_worse[file]["best"]
    worst = best_worse[file]["worst"]
    print(f"Best: QID: {best[0]} with MAP {best[1]}")
    print(f"query text: {queries[best[0]]}")
    print(f"Worst: QID: {worst[0]} with MAP {worst[1]}")
    print(f"query text: {queries[worst[0]]}")


