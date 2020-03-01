import os
import matplotlib.pyplot as plt
import numpy as np
import json


os.makedirs("plots", exist_ok=True)

curr_path = os.path.join(os.getcwd(), "json_files/")

json_files = [file for file in os.listdir(curr_path) if file.endswith(".json") and file.startswith("LSI")]
#print(json_files)
scores = dict()

for file in json_files:
    with open(os.path.join("json_files", file)) as f:
        scores_dict = json.load(f)
    metrics = {}
    avg_map_all = np.mean([scores_dict[query_scores]["map"] for query_scores in 
        scores_dict])
    avg_map_76_100 = np.mean([scores_dict[query_scores]["map"] for query_scores in 
        scores_dict if int(query_scores) in range(76, 101)])
    metrics["test"] = avg_map_76_100
    metrics["all"] = avg_map_all


    scores[file] = metrics



# nr topics size
test_scores = [scores[file]["test"] for file in json_files]
all_scores = [scores[file]["all"] for file in json_files]
nice_labels = [int(file.split("_")[1][:-5]) for file in json_files]

_, test_scores = zip(*sorted(zip(nice_labels, test_scores)))
nice_labels, all_scores = zip(*sorted(zip(nice_labels, all_scores)))

nice_labels = [str(i) for i in nice_labels]
    

plt.figure()
plt.bar(nice_labels, test_scores)
plt.xticks(rotation=50)
plt.title("Test queries: MAP for different number of topics")
plt.xlabel("Number of topics")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/LSI_test.png")


plt.figure()
plt.bar(nice_labels, all_scores)
plt.xticks(rotation=50)
plt.title("All queries: MAP for different number of topics")
plt.xlabel("Number of Topics")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/LSI_all.png")

