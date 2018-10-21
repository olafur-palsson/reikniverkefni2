
import torch
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as Optimizer
import numpy as np
from functools import reduce
from torch.autograd import Variable


learning_rate = 1e-3
dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

input_width, output_width = 102, 1
# hidden_layers_width = [500, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 55, output_width]
hidden_layers_width = [50, 50, 50, 50, 50]


def make_layers():
    last_width = input_width
    layers = []
    for width in hidden_layers_width:
        layers.append(nn.Linear(last_width, width))
        last_width = width
        # layers.append(nn.ReLU()) # uncomment for ReLU
    final = nn.Linear(last_width, output_width)
    layers.append(final)
    return layers

class BasicNetworkForTesting():

    def __init__(self):
        self.model = nn.Sequential(*make_layers())
        self.predictions = torch.empty((1), dtype = dtype, requires_grad=True)
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def run_decision(self, board_features):
        vector = board_features
        prediction = self.model(board_features)
        self.predictions = torch.cat((self.predictions, prediction))

    def predict(self, board_features):
        with torch.no_grad():
            return self.model(board_features)

    def get_reward(self, reward):
        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype) * reward
        loss = (self.predictions - y).pow(2).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.predictions - torch.ones((episode_length), dtype=dtype) * torch.mean(self.predictions))
        print("   Last prediction (optimally 1 or -1) " + str(float(self.predictions[episode_length - 1])) + "  " + str(y[0]))
        print("")
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()

class NeuralNetwork(nn.Module):

    network = []
    predictions = torch.zeros(1, dtype = dtype, requires_grad=True)

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.network, self.final = make_layers()
        print(self.network[0].parameters())
        parameters = []

        print("Parameters of each layer")
        i = 0
        for layer in self.network:
            # in between if i % 2 == 1: continue # uncomment for  ReLU
            req_grad, data = layer.parameters()
            print("")
            print("Layer " + str(i))
            i = i + 1
            print(data)
            parameters.append(data)
        bool,final_data = self.final.parameters()
        parameters.append(data)
        self.optimizer = torch.optim.Adam(parameters, learning_rate)

    def forward(self, input):
        vector = input
        for layer in self.network:
            vector = layer(vector)
        return self.final(vector)

    def run_decision(self, board_features):
        vector = board_features
        prediction = self(board_features)
        self.predictions = torch.cat((self.predictions, prediction))

    def predict(self, board_features):
        with torch.no_grad():
            prediction = self(board_features)
            return prediction

    def get_reward(self, reward):
        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype) * reward
        loss = torch.abs(self.predictions - y).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.predictions - torch.ones((episode_length), dtype=dtype) * torch.mean(self.predictions))
        print("   Last prediction (optimally 1 or -1) " + str(float(self.predictions[episode_length - 1])) + "  " + str(y[0]))
        print("")
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()
