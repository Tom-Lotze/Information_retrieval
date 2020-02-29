# -*- coding: utf-8 -*-
# python 3
# @Author: TomLotze
# @Date:   2020-02-29 22:35:11
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-02-29 23:42:13

import os
import json
import numpy as np
from collections import defaultdict


# variance only in terms of map



curr_path = os.path.join(os.getcwd(), "json_files/")

json_files = [file for file in os.listdir(curr_path) if file.endswith(".json")]
nr_files = len(json_files)
#print(json_files)
MAPS = {}

for file in json_files:
    with open(os.path.join("json_files", file)) as f:
        scores_dict = json.load(f)
    MAPS[file] = scores_dict


query_ids = scores_dict.keys()
#print(f"query_ids: {query_ids}")

variances = {qid:np.var(np.array([MAPS[file][qid]['map'] for file in json_files])) for qid in query_ids}


sorted_variances = sorted(variances.items(), key=lambda kv: -kv[1])

print("Queries with the highest variance between the retrieval models:")
for i in range(5):
    print(f"{i}: query {int(sorted_variances[i][0])} with variance {sorted_variances[i][1]:.5f}")




# output 5 query IDs with the highest variance and their variance