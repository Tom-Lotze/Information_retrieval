import pandas as pd
import numpy as np
import torch
import sys

sys.path.append('.')
sys.path.append('..')


def dcg_at_k(sorted_labels, k):
    if k > 0:
        k = min(sorted_labels.shape[0], k)
    else:
        k = sorted_labels.shape[0]
    denom = 1./np.log2(np.arange(k)+2.)
    nom = 2**sorted_labels-1.
    dcg = np.sum(nom[:k]*denom)
    return dcg


def ndcg_at_k(sorted_labels, ideal_labels, k):
    return dcg_at_k(sorted_labels, k) / dcg_at_k(ideal_labels, k)


def fast_dcg_at_k_(y, denom_order):

    discount = 1 / torch.log2(denom_order.float()+2)

    gain = 2**y - 1

    dcg = torch.sum(discount*gain)

    return dcg


def fast_delta_dcg_at_k_(y, denom_order):

    k_t = denom_order.float().unsqueeze(1)
    y = y.unsqueeze(1)
    discount = (1 / torch.log2(k_t + 2) -
                (1 / torch.log2(k_t.t() + 2)))

    gain = torch.pow(2.0, y) - \
        torch.pow(2.0, y.t())

    dcg = discount * gain

    return(abs(dcg))


def dcg_at_k_(sorted_labels):

    k = sorted_labels.shape[0]
    discount = 1 / torch.log2(torch.arange(k).float()+2)
    gain = 2 ** sorted_labels - 1
    dcg = torch.sum(discount*gain)
    return dcg


def fast_delta_ndcg_at_k_(y, denom_order, ideal_labels):
    return fast_delta_dcg_at_k_(y, denom_order) / dcg_at_k_(ideal_labels)


def delta_dcg_at_k_(sorted_labels):

    k = sorted_labels.shape[0]
    k_t = torch.arange(k).float().unsqueeze(1)

    discount = (1 / torch.log2(k_t + 2) -
                (1 / torch.log2(k_t.t() + 2)))
    # print(sorted_labels)
    gain = torch.pow(2.0, sorted_labels.unsqueeze(1)) - \
        torch.pow(2.0, sorted_labels.unsqueeze(1).t())

    # print("gain", gain)
    # print("dicount", discount)
    dcg = discount * gain

    return(abs(dcg))


def dcg_at_k_(sorted_labels):

    k = sorted_labels.shape[0]
    discount = 1 / torch.log2(torch.arange(k).float()+2)
    gain = 2 ** sorted_labels - 1
    dcg = torch.sum(discount*gain)
    return dcg


def unsorted_dcg_at_k_(y, denom_order):

    discount = 1 / torch.log2(denom_order.float()+2)
    gain = 2 ** y - 1

    # print(f"un_disc {discount} \n un_gain {gain}")
    dcg = torch.sum(discount*gain)

    # for i in range(y.shape[0]):
    #     print(
    #         f'{round(discount[i].item(),3)} * {round(gain[i].item(),3)} = {round((discount[i] * gain[i]).item(),3)}\n')

    return dcg


def delta_ndcg_at_k_(sorted_labels, ideal_labels):
    return delta_dcg_at_k_(sorted_labels) / dcg_at_k_(ideal_labels)


if __name__ == "__main__":
    import evaluate as evl
    x = torch.Tensor(6, 501)

    y = torch.Tensor([[1.],
                      [4.],
                      [1.],
                      [0.],
                      [2.],
                      [0.]])

    # y = torch.Tensor([[1.],
    #                   [2.],
    #                   [3.],
    #                   [4.],
    #                   [5.],
    #                   [6.],
    #                   [7.],
    #                   [8.],
    #                   [9.],
    #                   [10.],
    #                   [11.],
    #                   [12.],
    #                   [13.]])

    # scores = torch.Tensor([[0.4754],
    #                        [0.4426],
    #                        [0.5051],
    #                        [0.4982],
    #                        [0.4942],
    #                        [0.4471]])

    scores = torch.Tensor([[0.4754],
                           [0.4426],
                           [0.5051],
                           [0.4982],
                           [0.4942],
                           [0.4471]])

    scores2 = torch.Tensor([[0.4471],
                            [0.4426],
                            [0.5051],
                            [0.4982],
                            [0.4942],
                            [0.4754]])

    y = y.squeeze()
    scores = scores.squeeze()

    _, sort_ind = scores.sort(descending=True)

    predicted_order = y[sort_ind]

    _, sort_ind2 = scores2.sort(descending=True, dim=0)

    predicted_order2 = y[sort_ind2]

    ideal_labels, _ = y.sort(descending=True, dim=0)

    rank_df = pd.DataFrame(
        {'scores': scores, 'doc': np.arange(scores.shape[0])})
    rank_df = rank_df.sort_values(
        'scores', ascending=False).reset_index(drop=True)
    rank_order = rank_df.sort_values('doc').index.values

    denom_order = torch.Tensor(rank_order)

    # print(unsorted_dcg_at_k_(y, denom_order).item())

    print('goal: ', abs(ndcg_at_k(predicted_order.squeeze().numpy(),
                                  ideal_labels.squeeze().numpy(), 0) - ndcg_at_k(predicted_order2.squeeze().numpy(),
                                                                                 ideal_labels.squeeze().numpy(), 0)))

    # print("y: ", y)
    # print("denom order: ", denom_order)
    # print("scores :", scores)
    print("fast :", fast_delta_ndcg_at_k_(y.float(), denom_order,
                                          ideal_labels.squeeze().float())[0, 5])

    # print('wrong indices: ', delta_ndcg_at_k_(predicted_order.squeeze().float(),
    #                                           ideal_labels.squeeze().float()))

    # compute delta ndcg in same shape as lambda_i

    ind = list(range(x.shape[0]))
    index_mat = [(i, j) for i in ind for j in ind]

    true_ndcg = evl.evaluate_labels_scores(
        y.squeeze().numpy(), scores.squeeze().numpy())['ndcg']

    delta_ndcg = []

    for (i, j) in index_mat:
        new_labels = y.clone()
        # temp_label = new_labels[j].clone()
        # new_labels[j] = new_labels[i].clone()
        # new_labels[i] = temp_label
        new_scores = scores.clone()

        temp_var = new_scores[j].clone()
        new_scores[j] = new_scores[i].clone()
        new_scores[i] = temp_var

        new_ndcg = evl.evaluate_labels_scores(
            new_labels.squeeze().numpy(), new_scores.detach().squeeze().numpy())

        delta = abs(true_ndcg - new_ndcg['ndcg'])
        delta_ndcg.append(delta)

    # results_order = evl.evaluate_labels_scores(labels, scores)
    delta_ndcg = torch.Tensor(delta_ndcg).view(
        x.shape[0], x.shape[0])

    print('slow method: ', delta_ndcg[0, 5])
