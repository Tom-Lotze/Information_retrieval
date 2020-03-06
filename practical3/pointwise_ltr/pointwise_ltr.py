import torch
import torch.nn as nn
import numpy as np
import os
import sys
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
        #out = self.softmax(x)

        return out


    def evaluate_on_validation(self, data):
        with torch.no_grad():
            averages = []
            predictions_list = []

            validation_data_generator = DataLoader(data.validation, batch_size=2042, shuffle=False, drop_last=False)

            for step, (x_valid, y_valid) in enumerate(
                            validation_data_generator):
                x_valid, y_valid = x_valid.float().to(self.device), Variable(y_valid).to(self.device)
                logits = self.forward(x_valid)
                predictions = np.argmax(logits, axis=1)
                predictions_list.extend(list(predictions))

            results = evl.evaluate(data.validation, np.array(predictions_list), print_results=True)

        return results



    def evaluate_on_test(self, x_test, y_test):
        model.eval()
        pass



    def accuracy(self, predictions, labels):
        batch_size = labels.shape[0]
        predictions = predictions.argmax(dim=1)
        total_correct = torch.sum(predictions == labels).item()
        accuracy = total_correct / batch_size
        # print(f"total_correct: {total_correct}, batch_size: {batch_size}")

        return accuracy


    def ndcg(self, predictions, labels, k=10):
        return evl.ndcg_at_k(sorted_labels, ideal_labels, k)



# helper functions
def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)

# train function
def train(data):
    #model = Pointwise(data.num_features, [512, 256, 128, 64, 8])
    n_hidden = [512, 128, 8]
    model = Pointwise(data.num_features, n_hidden, n_outputs=5)
    model.apply(weights_init)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    nr_epochs = 40
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training_data_generator = DataLoader(data.train, batch_size=512, shuffle=True, drop_last=True, num_workers = 4)

    model.to(device)

    training_losses = []
    validation_results = {}

    for epoch in range(nr_epochs):
        print(f"Epoch: {epoch}")
        model.train()

        for step, (x, y) in enumerate(training_data_generator):
            x, y = x.float().to(device), Variable(y).to(device)

            # reset the optimizer and perform forward pass
            optimizer.zero_grad()
            predictions = model(x)

            # compute the loss and backpropagate
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            training_losses.append(loss_item)
            if step % 100 == 0:
                print(f"Step: {step}: Loss: {loss_item:.4f}")



        # save mode
        filename = f"./pointwise_ltr/models/pointwise_{n_hidden}_{epoch}_{learning_rate}.pt"
        torch.save(model.state_dict(), filename)
        print(f"Model is saved as {filename}")

        # run on validation set
        model.eval()
        results_validation = model.evaluate_on_validation(data)
        validation_results[epoch] = results_validation


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)


    # import the data
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    os.makedirs("pointwise_ltr/models", exist_ok=True)


    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' % data.validation.num_queries())
    print('Number of documents in validation set: %d' % data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())


    train(data)



