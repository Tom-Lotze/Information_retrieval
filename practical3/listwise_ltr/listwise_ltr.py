import torch
import torch.nn as nn
import numpy as np


class Listwise(nn.Module):
    """ Listwise LTR model """

    def __init__(self, input_size=501):
        """ Initialize model """
        super(Listwise, self).__init__()

        self.nnet = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20,1),
            nn.ReLU()
        )

    def forward(self, x):
        """ Forward pass """
        return self.nnet(x)


if __name__ == "__main__":
    pass
