# -*- coding: utf-8 -*-
# python 3
# @Author: TomLotze
# @Date:   2020-02-29 22:35:11
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-02-29 23:54:13

import os
import json
import numpy as np
from collections import defaultdict

import read_ap

curr_path = os.path.join(os.getcwd(), "json_files/")

json_files = [file for file in os.listdir(curr_path) if file.endswith(".json")]
nr_files = len(json_files)
MAPS = {}

for file in json_files:
    with open(os.path.join("json_files", file)) as f:
        scores_dict = json.load(f)
    MAPS[file] = scores_dict

query_ids = scores_dict.keys()
variances = {qid:np.var(np.array([MAPS[file][qid]['map'] for file in json_files])) for qid in query_ids}
sorted_variances = sorted(variances.items(), key=lambda kv: -kv[1])

qrels, queries = read_ap.read_qrels()


print("Queries with the highest variance between the retrieval models:")
for i in range(5):
    qid = sorted_variances[i][0]
    print(f"{i+1}: query {int(qid)} with variance in MAP of {sorted_variances[i][1]:.5f}")
    print(f"text: {queries[qid]}")

