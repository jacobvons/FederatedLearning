import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LinearRegression(nn.Module):

    def __init__(self, input_size, depth):
        """
        Linear Regression module

        :param input_size: Dimensionality of the input array
        :param depth: Number of trainable layers (ones need grads and biases)
        """
        super().__init__()
        self.depth = depth
        self.linear = nn.Linear(input_size, 1)
        self.layers = [self.linear]

    def forward(self, x):
        """
        Perform one pass forward step.

        :param x: Input tensor
        :return: Tensor after one layer of fully connected neural network
        """
        y = self.linear(x)
        return y


class MLPRegression(nn.Module):

    def __init__(self, input_size, hidden_size, target_size, depth):
        """
        Multi-layer perceptron regressor

        :param input_size: Size of input tensor
        :param hidden_size: Size of hidden layer
        :param target_size: Size of output tensor
        :param depth: Number of layers that needs gradients and biases
        """
        super().__init__()
        self.depth = depth
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, target_size)
        self.layers = [self.linear1, self.linear2]

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        return y
