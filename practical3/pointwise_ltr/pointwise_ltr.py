import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import pickle as pkl

from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('..')
sys.path.append(".")
import dataset
#import ranking
import evaluate as evl


class Pointwise(nn.Module):
    """
    Model class for the pointwise ranking model
    """
    def __init__(self, n_inputs, n_hidden, n_outputs=5):
        """
        Model that takes an input feature vector x and tries to predict the relevance score
        Input arguments:
        - n_inputs: the dimensionality of the feature vector
        - n_hidden: list of dimensions for the hidden layers
        - n_outputs: output dimensionality
        """
        super(Pointwise, self).__init__()

        # initialize parameters
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # set non-linearity
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # add linear layers
        layer_list = [nn.Linear(n_inputs, n_hidden[0])]
        if len(n_hidden) > 1:
            for i, n_hid in enumerate(n_hidden[1:], start=1):
                layer_list.append(nn.Linear(n_hidden[i-1], n_hid))
        layer_list.append(nn.Linear(n_hidden[-1], n_outputs))

        self.layers = nn.ModuleList(layer_list)

        print(self.layers)


    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)

        out = self.layers[-1](x)

        return out


    def evaluate_on_validation(self, data):
        with torch.no_grad():
            predictions_list = []

            validation_data_generator = DataLoader(data.validation, batch_size=2042, shuffle=False, drop_last=False)

            for step, (x_valid, y_valid) in enumerate(
                            validation_data_generator):
                x_valid, y_valid = x_valid.float().to(self.device), Variable(y_valid).to(self.device)
                logits = self.forward(x_valid)
                predictions = np.argmax(logits, axis=1)
                predictions_list.extend(list(predictions))

            results = evl.evaluate(data.validation, np.array(predictions_list), print_results=False)

        return results



    def evaluate_on_test(self, data, FLAGS):
        with torch.no_grad():
            predictions_list = []

            test_data_generator = DataLoader(data.test, batch_size=2042, shuffle=False, drop_last=False)

            for step, (x_test, y_test) in enumerate(
                            test_data_generator):
                x_test, y_test = x_test.float().to(self.device), Variable(y_test).to(self.device)
                logits = self.forward(x_test)
                predictions = np.argmax(logits, axis=1)
                predictions_list.extend(list(predictions))

            # save predictions list
            if FLAGS.save_pred:
                with open("pointwise_LTR/predictions.pt", "wb") as writer:
                    pkl.dump(predictions_list, writer)

            results = evl.evaluate(data.test, np.array(predictions_list), print_results=False)

        return results


    def ndcg(self, predictions, labels, k=10):
        return evl.ndcg_at_k(sorted_labels, ideal_labels, k)


# helper functions
def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


# train function
def train(data, FLAGS):
    # initialize model
    n_hidden = [int(n_h) for n_h in FLAGS.hidden_units.split(",")]
    model = Pointwise(data.num_features, n_hidden, n_outputs=5)
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    training_data_generator = DataLoader(data.train, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True, num_workers = 4)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # init results lists
    training_losses = []
    validation_results = {}
    ndcg_per_epoch = {}
    filename_validation_results = f"./pointwise_ltr/json_files/pointwise_{n_hidden}_{FLAGS.learning_rate}_Adam.json"
    filename_test_results = f"./pointwise_ltr/json_files/pointwise_TEST_{n_hidden}_{FLAGS.learning_rate}_Adam.json"
    figure_name = f"{n_hidden}_{FLAGS.learning_rate}_Adam.png"

    model.to(device)
    overall_step = 0

    for epoch in range(FLAGS.max_epochs):
        print(f"Epoch: {epoch+1}")
        ndcg_per_epoch[epoch] = {}
        # iterate over batches
        for step, (x, y) in enumerate(training_data_generator):
            model.train()
            x, y = x.float().to(device), Variable(y).to(device)

            # reset the optimizer and perform forward pass
            optimizer.zero_grad()
            predictions = model(x)

            # compute the loss and backpropagate
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            loss_item = loss.item()

            if overall_step % 100 == 0:
                print(f"Batch: {overall_step}: Loss: {loss_item:.4f}")

            if overall_step % FLAGS.valid_each == 0:
                # run on validation set
                model.eval()
                results_validation = model.evaluate_on_validation(data)
                validation_results[overall_step] = results_validation
                ndcg_per_epoch[epoch][step] = results_validation["ndcg"][0]
                training_losses.append(loss_item)

            overall_step += 1

        # save model
        if epoch % 3 == 0 and epoch != 0 and FLAGS.save:
            filename_model = f"./pointwise_ltr/models/pointwise_{n_hidden}_{epoch}_{FLAGS.learning_rate}_Adam.pt"
            torch.save(model.state_dict(), filename_model)
            print(f"Model is saved as {filename_model}")

        # early stopping
        if epoch > 0:
            mean_prev_epoch = np.mean(list(ndcg_per_epoch[epoch-1].values()))
            mean_curr_epoch = np.mean(list(ndcg_per_epoch[epoch].values()))
            print(mean_prev_epoch, mean_curr_epoch)
            if mean_curr_epoch <= mean_prev_epoch + FLAGS.early_stopping_threshold:
                print("Early stopping condition satistied, stopping training")
                break

    # save results
    if FLAGS.plot:
        plot_loss_ndcg(validation_results, training_losses, figure_name)

    if FLAGS.save:
        with open(filename_validation_results, "w") as writer:
            json.dump(validation_results, writer, indent=1)
        with open(filename_test_results, "w") as writer:
            json.dump(model.evaluate_on_test(data, FLAGS), writer, indent=1)
        print(f"Results are saved in the json_files folder")


def plot_loss_ndcg(ndcg, loss, figname):
    ndcg_values = [i["ndcg"][0] for i in ndcg.values()]
    x_labels = list(ndcg.keys())


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Batch (924 batches per epoch)')
    ax1.set_ylabel('nDCG', color=color)
    ax1.plot(x_labels, ndcg_values, color=color, label="nDCG")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc=0)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Training loss', color=color)
    ax2.plot(x_labels, loss, color=color, label="Training loss")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc=1)

    plt.title("nDCG and training loss for Pointwise LTR")
    plt.tight_layout()
    plt.savefig(f"pointwise_ltr/figures/{figname}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type = str, default = "256, 10", help='Comma separated list of unit numbers in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type = int, default = 40, help='Max number of epochs')
    parser.add_argument('--batch_size', type = int, default = 512, help='Batch size')
    parser.add_argument("--save", type=int, default=1, help="Either 1 or 0 (bool) to save the model and results")
    parser.add_argument("--plot", type=int, default=0, help="Either 1 or 0 (bool) to create a plot of ndcg and loss")
    parser.add_argument("--valid_each", type=int, default=100, help="Run the model on the validation set every x steps")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Minimal difference in ndcg on validation set between epochs to continue training")
    parser.add_argument("--save_pred", type = int, default=0, help="Boolean (0, 1) whether to save the predictions on the test set")

    # set configuration in FLAGS parameter
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.save = bool(FLAGS.save)
    FLAGS.plot = bool(FLAGS.plot)
    FLAGS.save_pred = bool(FLAGS.save_pred)


    # set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # import the data
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    # create instances specific for pointwise model
    data.train = dataset.Pointwise_fold(data.train)
    data.validation = dataset.Pointwise_fold(data.validation)
    data.test = dataset.Pointwise_fold(data.test)

    # create necessary datasets
    os.makedirs("pointwise_ltr/models", exist_ok=True)
    os.makedirs("pointwise_ltr/json_files", exist_ok=True)
    os.makedirs("pointwise_ltr/figures", exist_ok=True)

    # print information about dataset, if needed
    if False:
        print('Number of features: %d' % data.num_features)
        print('Number of queries in training set: %d' % data.train.num_queries())
        print('Number of documents in training set: %d' % data.train.num_docs())
        print('Number of queries in validation set: %d' % data.validation.num_queries())
        print('Number of documents in validation set: %d' % data.validation.num_docs())
        print('Number of queries in test set: %d' % data.test.num_queries())
        print('Number of documents in test set: %d' % data.test.num_docs())

    # train the model given the current hyperparameters
    train(data, FLAGS)



