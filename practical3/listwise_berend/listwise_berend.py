import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torch.autograd import Variable
import torch
import argparse
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append('..')
sys.path.append('.')


class Listwise(nn.Module):
    """ Listwise LTR model """

    def __init__(self, input_dim,  n_hidden=256,  output_dim=1):
        """
        Initialize model
        input_dim: dimensionality of document feature vector
        n_hidden: dimensionality of hidden layer
        n_outputs: output dimensionality
        """
        super(Listwise, self).__init__()

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_batch):
        """
        Forward pass
        - x: batch of document pairs B x (doc_i, doc_j)
        - out: batch of document scores: B x 1
        """

        h_i = self.sigmoid(self.fc1(x_batch))
        s_i = self.sigmoid(self.fc2(h_i))

        return s_i

    def single_foward(self, x_batch):
        """ Single forward pass"""
        h_i = self.sigmoid(self.fc1(x_batch))
        s_i = self.sigmoid(self.fc2(h_i))

        return s_i

    def evaluate_on_validation(self, data):
        """ evaluate on validation """
        valid_data = data.validation
        with torch.no_grad():
            valid_scores = self.single_foward(
                torch.Tensor(valid_data.feature_matrix))
            valid_scores = valid_scores.numpy().squeeze()
            results = evl.evaluate(valid_data, valid_scores)
        return results

    def evaluate_on_test(self, data):
        """ Evaluate on test set """
        test_data = data.test
        with torch.no_grad():
            test_scores = self.single_foward(
                torch.Tensor(test_data.feature_matrix))
            test_scores = test_scores.numpy().squeeze()
            results = evl.evaluate(test_data, test_scores)
        return results


def delta_dcg_at_k(y, denom_order):

    k_t = denom_order.float().unsqueeze(1)
    y = y.unsqueeze(1)
    discount = (1 / torch.log2(k_t + 2) -
                (1 / torch.log2(k_t.t() + 2)))

    gain = torch.pow(2.0, y) - \
        torch.pow(2.0, y.t())

    dcg = discount * gain

    return(abs(dcg))


def dcg_at_k(sorted_labels):

    k = sorted_labels.shape[0]
    discount = 1 / torch.log2(torch.arange(k).float()+2)
    gain = 2 ** sorted_labels - 1
    dcg = torch.sum(discount*gain)
    return dcg


def delta_ndcg_at_k(y, denom_order, ideal_labels):
    return delta_dcg_at_k(y, denom_order) / dcg_at_k(ideal_labels)


def weights_init(model):
    """ Initialize weights """
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


def train(data, FLAGS):

    n_hidden = FLAGS.hidden_units
    n_epochs = FLAGS.max_epochs
    metric = FLAGS.metric

    model = Listwise(data.num_features, n_hidden, 1)
    model.apply(weights_init)

    train_dataset = dataset.ListDataSet(data.train)
    train_dl = DataLoader(train_dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    validation_results = {}
    ndcg_per_epoch = {}

    filename_validation_results = (f"./listwise_berend/json_files/"
                                   f"pairwise_{n_hidden}_"
                                   f"{FLAGS.learning_rate}.json")
    filename_test_results = (f"./listwise_berend/json_files/"
                             f"listwise_TEST_{n_hidden}_"
                             f"{FLAGS.learning_rate}.json")
    figure_name = f"{n_hidden}_{FLAGS.learning_rate}.png"

    overall_step = 0

    for epoch in range(n_epochs):
        ndcg_per_epoch[epoch] = {}

        with tqdm(total=len(train_dl)) as t:
            t.set_description(f'Epoch: {epoch+1}/{n_epochs}')

            for step, (x_batch, y_batch) in enumerate(train_dl):

                optimizer.zero_grad()

                # ignore batch if only one doc (no pairs)
                if x_batch.shape[1] == 1:
                    continue

                # prepare batches
                x_batch = x_batch.float().squeeze()
                y_batch = y_batch.float().t()

                y_batch.sort(descending=True)

                # Define factoriaztion
                S = y_batch - y_batch.t()

                S[S > 0] = 1
                S[S == 0] = 0
                S[S < 0] = -1

                # compute document scores
                scores = model(x_batch)

                # get document score differences
                diff_mat = torch.sigmoid(torch.add(scores, -scores.t()))

                # calculate lambda ij
                lambda_ij = (
                    1 * ((1/2) * (1 - S) - diff_mat))

                if overall_step % FLAGS.valid_each == 0:
                    # print(lambda_ij)
                    model.eval()
                    valid_results = model.evaluate_on_validation(data)
                    validation_results[overall_step] = valid_results
                    ndcg_per_epoch[epoch][step] = valid_results['ndcg'][0]
                    t.set_postfix_str(
                        (f'ndcg: '
                         f' {valid_results["ndcg"][0]: 3f}'))

                if metric == "NDCG":
                    clone_scores = scores.clone().detach()
                    # inspired by https://github.com/haowei01/pytorch-examples/blob/6c217bb995db6bc33a13f4828035f51365ed0eb9/ranking/LambdaRank.py#L57
                    rank_df = pd.DataFrame(
                        {'scores': clone_scores, 'doc': np.arange(clone_scores.shape[0])})
                    rank_df = rank_df.sort_values(
                        'scores', ascending=False).reset_index(drop=True)
                    rank_order = rank_df.sort_values('doc').index.values

                    if (y_batch == 0).all():
                        continue

                    denom_order = torch.Tensor(rank_order)
                    ideal_labels, _ = y_batch.sort(descending=True, dim=0)
                    delta_ndcg = delta_ndcg_at_k(
                        y_batch.squeeze().float(), denom_order, ideal_labels.squeeze().float())

                    lambda_ij *= delta_ndcg

                elif metric == "ERR":
                    raise NotImplementedError
                else:
                    raise NotImplementedError

                lambda_i = lambda_ij.sum(dim=1)

                scores.squeeze().backward(lambda_i)
                # print(model.fc1.weight.grad)
                # optimizer step
                optimizer.step()

                t.update()

                overall_step += 1

            if epoch % 1 == 0 and FLAGS.save:
                filename_model = (
                    f"./listwise_berend/models/listwise_{n_hidden}"
                    f"_{epoch}_{FLAGS.learning_rate}.pt")
                torch.save(model.state_dict(), filename_model)
                print(f"Model is saved as {filename_model}")

            # early stopping
            if epoch > 0:
                mean_prev_epoch = np.mean(
                    list(ndcg_per_epoch[epoch-1].values()))
                mean_curr_epoch = np.mean(list(ndcg_per_epoch[epoch].values()))
                # print(mean_prev_epoch, mean_curr_epoch)
                if (mean_curr_epoch <= mean_prev_epoch +
                        FLAGS.early_stopping_threshold):
                    print("Early stopping condition satistied, "
                          "stopping training")
                    break

    # save results
    if FLAGS.plot:
        plot_loss_ndcg(validation_results, figure_name)

    if FLAGS.save:
        with open(filename_validation_results, "w") as writer:
            json.dump(validation_results, writer, indent=1)
        with open(filename_test_results, "w") as writer:
            json.dump(model.evaluate_on_test(data), writer, indent=1)
        print(f"Results are saved in the json_files folder")


def plot_loss_ndcg(ndcg, figname):
    ndcg_values = [i["ndcg"][0] for i in ndcg.values()]
    x_labels = list(ndcg.keys())

    plt.figure()
    color = 'tab:red'
    plt.xlabel('Batch (1 query per batch)')
    plt.ylabel('nDCG')
    plt.plot(x_labels, ndcg_values, label="nDCG")
    plt.tick_params(axis='y')
    plt.legend()

    plt.title("nDCG and training loss for NDCG-LambdaRank")
    plt.tight_layout()
    plt.savefig(f"listwise_berend/figures/{figname}")


if __name__ == "__main__":
    # currently importing dataset here as pep8 would move it to the
    # top messing up the sys.path code
    import dataset
    import evaluate as evl

    np.random.seed(42)
    torch.manual_seed(42)
    # # load and read data
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type=int, default=256,
                        help=('Integer specifying number of hidden units in '
                              'hidden layer'))
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int,
                        default=10, help='Max number of epochs')
    parser.add_argument("--save", type=int, default=1,
                        help=("Either 1 or 0 (bool) to save"
                              " the model and results"))
    parser.add_argument("--plot", type=int, default=0,
                        help=("Either 1 or 0 (bool) to create a plot "
                              "of ndcg and loss"))
    parser.add_argument("--valid_each", type=int, default=100,
                        help=("Run the model on the validation "
                              "set every x steps"))
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help=("Minimal difference in ndcg on validation "
                              "set between epochs to continue training"))
    parser.add_argument("--save_pred", type=int, default=0,
                        help=("Boolean (0, 1) whether to save the "
                              "predictions on the test set"))
    parser.add_argument("--metric", type=str, default='NDCG',
                        help=("Choose metric to optimize."
                              "NDCG or ERR"))

    # set configuration in FLAGS parameter
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.save = bool(FLAGS.save)
    FLAGS.plot = bool(FLAGS.plot)
    FLAGS.save_pred = bool(FLAGS.save_pred)

    # # print data stats
    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' %
          data.validation.num_queries())
    print('Number of documents in validation set: %d' %
          data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())

    # create necessary datasets
    os.makedirs("listwise_berend/models", exist_ok=True)
    os.makedirs("listwise_berend/json_files", exist_ok=True)
    os.makedirs("listwise_berend/figures", exist_ok=True)

    # train model
    train(data, FLAGS)
