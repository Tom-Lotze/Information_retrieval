import json
import os

with open("pointwise_ltr/json_files/Best model_256_10-0.001_adam2.json") as f:
    scores = json.load(f)

print(scores)

for (metric, [score, std]) in scores.items():
    print(f"{metric} & {score:.3f} & {std:.3f}\\\\")

