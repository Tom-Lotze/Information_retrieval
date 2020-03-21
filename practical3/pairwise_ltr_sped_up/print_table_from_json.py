import json
import os

with open("pairwise_ltr_sped_up/json_files/pairwise_TEST_128_0.005.json") as f:
    scores = json.load(f)

print(scores)

for (metric, [score, std]) in scores.items():
    print(f"{metric} & {score:.3f} & {std:.3f}\\\\")

