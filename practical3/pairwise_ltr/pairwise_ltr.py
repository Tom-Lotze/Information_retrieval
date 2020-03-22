import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
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

import dataset
import evaluate as evl


class RankNet(nn.Module):
    """ Pairwise LTR model """

    def __init__(self, input_dim,  n_hidden=128,  output_dim=1):
        """
        Initialize model
        input_dim: dimensionality of document feature vector
        n_hidden: dimensionality of hidden layer
        n_outputs: output dimensionality
        """
        super(RankNet, self).__init__()

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

        diff_mat = self.sigmoid(torch.add(s_i.t(), -s_i))

        return diff_mat

    def single_forward(self, x_batch):

        h_i = self.sigmoid(self.fc1(x_batch))
        s_i = self.sigmoid(self.fc2(h_i))

        return s_i

    def evaluate_on_validation(self, data):
        """ evaluate on validation """
        valid_data = data.validation
        with torch.no_grad():
            valid_scores = self.single_forward(
                torch.Tensor(valid_data.feature_matrix))
            valid_scores = valid_scores.numpy().squeeze()
            results = evl.evaluate(valid_data, valid_scores)
        return results

    def evaluate_on_test(self, data):
        """ Evaluate on test set """
        test_data = data.test
        with torch.no_grad():
            test_scores = self.single_forward(
                torch.Tensor(test_data.feature_matrix))
            test_scores_np = test_scores.numpy().squeeze()
            results = evl.evaluate(test_data, test_scores_np)
        return test_scores, results


def weights_init(model):
    """ Initialize weights """
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)

def err(scores, test_labels):

    R = test_labels[scores.sort(descending=True, dim=0).indices]

    r = torch.arange(R.shape[0]) + 1
    denom = 2 ** 4
    Ri = torch.Tensor((2 ** R - 1) / denom)

    prod = torch.cumprod(1 - Ri, dim=0) / (1-Ri)
    err = torch.sum(Ri * prod / r)

    return err


def train(data, FLAGS):

    n_hidden = FLAGS.hidden_units
    n_epochs = FLAGS.max_epochs

    model = RankNet(data.num_features, n_hidden, 1)
    model.apply(weights_init)

    train_dataset = dataset.ListDataSet(data.train)
    train_dl = DataLoader(train_dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    loss_function = nn.BCELoss(reduction='mean')

    # training_losses = []
    validation_results = {}
    ndcg_per_epoch = {}

    filename_validation_results = f"./pairwise_ltr/json_files/pairwise_{n_hidden}_{FLAGS.learning_rate}.json"
    filename_test_results = f"./pairwise_ltr/json_files/pairwise_TEST_{n_hidden}_{FLAGS.learning_rate}.json"
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

                # squeeze batch
                x_batch = x_batch.float().squeeze()
                y_batch = y_batch.float().t()

                labels_mat = y_batch.t() - y_batch

                labels_mat[labels_mat > 0] = 1
                labels_mat[labels_mat == 0] = (1/2)
                labels_mat[labels_mat < 0] = 0

                diff_scores = model(x_batch)

                loss = loss_function(diff_scores, labels_mat)

                if overall_step % FLAGS.valid_each == 0:
                    model.eval()
                    valid_results = model.evaluate_on_validation(data)
                    validation_results[overall_step] = valid_results
                    ndcg_per_epoch[epoch][step] = valid_results['ndcg'][0]

                    t.set_postfix_str(
                        f'loss: {loss.mean().item():3f} ndcg: {valid_results["ndcg"][0]:3f}')

                # backward pass
                loss.backward()

                # optimizer step
                optimizer.step()

                t.update()

                overall_step += 1

            if epoch % 1 == 0 and FLAGS.save:
                filename_model = (
                    f"./pairwise_ltr/models/pairwise_{n_hidden}_{epoch}_{FLAGS.learning_rate}.pt")
                torch.save(model.state_dict(), filename_model)
                print(f"Model is saved as {filename_model}")

            # early stopping
            if epoch > 0:
                mean_prev_epoch = np.mean(
                    list(ndcg_per_epoch[epoch-1].values()))
                mean_curr_epoch = np.mean(list(ndcg_per_epoch[epoch].values()))

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
            json.dump(model.evaluate_on_test(data)[1], writer, indent=1)
        print(f"Results are saved in the json_files folder")

    # if FLAGS.save:
    #     with open(filename_validation_results, "w") as writer:
    #         json.dump(validation_results, writer, indent=1)
    #     with open(filename_test_results, "w") as writer:
    #         json.dump(model.evaluate_on_test(data), writer, indent=1)
    #     print(f"Results are saved in the json_files folder")


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

    plt.title("nDCG for Pairwise LTR")
    plt.tight_layout()
    plt.savefig(f"pairwise_ltr/figures/{figname}")


if __name__ == "__main__":

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
                        default=0.005, help='Learning rate')
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
    os.makedirs("pairwise_ltr/models", exist_ok=True)
    os.makedirs("pairwise_ltr/json_files", exist_ok=True)
    os.makedirs("pairwise_ltr/figures", exist_ok=True)

    # train model
    train(data, FLAGS)
