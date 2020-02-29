import os
import matplotlib.pyplot as plt
import numpy as np
import json


curr_path = os.path.join(os.getcwd(), "json_files/")

json_files = [file for file in os.listdir(curr_path) if file.endswith(".json") and file.startswith("benchmark_gensim_164557")]
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



nice_file_labels = [filename[24:-11] for filename in json_files]



test_scores_ = [scores[file]["test"] for file in json_files]
all_scores = [scores[file]["all"] for file in json_files]

# vocab size
test_scores_vocab = [scores[file]["test"] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
all_scores_vocab = [scores[file]["all"] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
nice_labels_vocab_test = [file[24:-11].split("_")[0]+" " for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
nice_labels_vocab_all = [file[24:-11].split("_")[0] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]

plt.figure()
plt.bar(nice_labels_vocab_test, test_scores_vocab, label="test queries")
plt.bar(nice_labels_vocab_all, all_scores_vocab, label = "all queries")
plt.xticks(rotation=50)
plt.title("MAP for different vocabulary sizes")
plt.xlabel("Vocabulary size of the model")
plt.ylabel("Mean Average Precision")
plt.legend()
plt.tight_layout()
plt.show()


# vector dim
test_scores_vecdim = [scores[file]["test"] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
all_scores_vecdim = [scores[file]["all"] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
nice_labels_vecdim_test = [file[24:-11].split("_")[0]+" " for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
nice_labels_vecdim_all = [file[24:-11].split("_")[0] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]

plt.figure()
plt.bar(nice_labels_vocab_test, test_scores_vocab, label="test queries")
plt.bar(nice_labels_vocab_all, all_scores_vocab, label = "all queries")
plt.xticks(rotation=50)
plt.title("MAP for different vocabulary sizes")
plt.xlabel("Vocabulary size of the model")
plt.ylabel("Mean Average Precision")
plt.legend()
plt.tight_layout()
plt.show()






