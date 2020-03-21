import listwise_ltr as listwise
import torch
import numpy as np
import sys
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')


if __name__ == "__main__":
    import evaluate as evl
    import dataset
    np.random.seed(42)
    torch.manual_seed(42)

    # load and read data
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    model = listwise.Listwise(input_dim=501, n_hidden=128)
    # model.load_state_dict(torch.load(
        # "./listwise_ltr/models/listwise_128_2_0.005.pt"))
    model.load_state_dict(torch.load(
        "./listwise_ltr/models/listwise_128_2_0.005_ERR.pt"))
    # perform forward pass
    model.eval()
    scores = model.evaluate_on_test(data)[0].squeeze()
    ERR_scores = []

    # compute ERR per query
    for qid in tqdm(np.arange(data.test.num_queries())):
        if evl.included(qid, data.test):
            s_i, e_i = data.test.doclist_ranges[qid:qid+2]
            q_scores = scores[s_i:e_i]
            if len(q_scores) < 2:
                continue
            q_labels = data.test.query_labels(qid)
            ERR_query = listwise.err(q_scores, q_labels)
            ERR_scores.append(ERR_query)

    print(f"Mean: {np.mean(ERR_scores):.4f}\nSTD: {np.std(ERR_scores):.4f}\nnr queries tested: {len(ERR_scores)}")
