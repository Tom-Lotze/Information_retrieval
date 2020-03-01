import os
import matplotlib.pyplot as plt
import numpy as np
import json


os.makedirs("plots", exist_ok=True)

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



# vocab size
test_scores_vocab = [scores[file]["test"] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
all_scores_vocab = [scores[file]["all"] for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]
nice_labels_vocab = [int(file[24:-11].split("_")[0]) for file in json_files if file[24:-11].split("_")[1:]==["300", "5"]]

_, test_scores_vocab = zip(*sorted(zip(nice_labels_vocab, test_scores_vocab)))
nice_labels_vocab, all_scores_vocab = zip(*sorted(zip(nice_labels_vocab, all_scores_vocab)))

nice_labels_vocab = [str(i) for i in nice_labels_vocab]
    

plt.figure()
plt.bar(nice_labels_vocab, test_scores_vocab)
plt.xticks(rotation=50)
plt.title("Test queries: MAP for different vocabulary sizes")
plt.xlabel("Vocabulary size of the model")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/vocab_test.png")


plt.figure()
plt.bar(nice_labels_vocab, all_scores_vocab)
plt.xticks(rotation=50)
plt.title("All queries: MAP for different vocabulary sizes")
plt.xlabel("Vocabulary size of the model")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/vocab_all.png")


# vector dim
test_scores_vecdim = [scores[file]["test"] for file in json_files if [file[24:-11].split("_")[0], file[24:-11].split("_")[2]]==["25000", "5"]]
all_scores_vecdim = [scores[file]["all"] for file in json_files if [file[24:-11].split("_")[0], file[24:-11].split("_")[2]]==["25000", "5"]]
nice_labels_vecdim = [file[24:-11].split("_")[1] for file in json_files if [file[24:-11].split("_")[0], file[24:-11].split("_")[2]]==["25000", "5"]]

_, test_scores_vecdim = zip(*sorted(zip(nice_labels_vecdim, test_scores_vecdim)))
nice_labels_vecdim, all_scores_vecdim = zip(*sorted(zip(nice_labels_vecdim, all_scores_vecdim)))

nice_labels_vecdim = [str(i) for i in nice_labels_vecdim]


plt.figure()
plt.bar(nice_labels_vecdim, test_scores_vecdim)
plt.xticks(rotation=50)
plt.title("Test queries: MAP for different embedding dimensions")
plt.xlabel("Embedding dimension")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/vectordim_test.png")

plt.figure()
plt.bar(nice_labels_vecdim, all_scores_vecdim)
plt.xticks(rotation=50)
plt.title("All queries: MAP for different embeddings dimensions")
plt.xlabel("Embedding dimension")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/vectordim_all.png")




# window size
test_scores_window = [scores[file]["test"] for file in json_files if file[24:-11].split("_")[:-1]==["25000", "300"]]
all_scores_window = [scores[file]["all"] for file in json_files if file[24:-11].split("_")[:-1]==["25000", "300"]]
nice_labels_window = [int(file[24:-11].split("_")[2]) for file in json_files if file[24:-11].split("_")[:2]==["25000", "300"]]

_, test_scores_window = zip(*sorted(zip(nice_labels_window, test_scores_window)))
nice_labels_window, all_scores_window = zip(*sorted(zip(nice_labels_window, all_scores_window)))

nice_labels_window = [str(i) for i in nice_labels_window]
    

plt.figure()
plt.bar(nice_labels_window, test_scores_window)
plt.xticks(rotation=50)
plt.title("Test queries: MAP for different window sizes")
plt.xlabel("Window size of the model")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/window_test.png")


plt.figure()
plt.bar(nice_labels_window, all_scores_window)
plt.xticks(rotation=50)
plt.title("All queries: MAP for different window sizes")
plt.xlabel("Window size of the model")
plt.ylabel("Mean Average Precision")
plt.tight_layout()
plt.savefig("./plots/window_all.png")




















