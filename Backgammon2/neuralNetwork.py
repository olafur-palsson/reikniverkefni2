
import torch
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as Optimizer
import numpy as np
from functools import reduce
from torch.autograd import Variable


learning_rate = 1e-3
dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

input_width, output_width = 464, 1
hidden_layers_width = [1000, 1000, output_width]
# hidden_layers_width = [150, 150]

all_width = 70

def make_layers():
    layers = []

    last_width = input_width

    """
    layers.append(nn.Linear(last_width, all_width))
    last_width = all_width
    for i in range(20):
        layers.append(nn.Linear(all_width, all_width))

    """

    for width in hidden_layers_width:
        layers.append(nn.Linear(last_width, width))
        last_width = width
        # layers.append(nn.ReLU6()) # uncomment for ReLU


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
        self.predictions = torch.cat((self.predictions, prediction.double()))

    def predict(self, board_features):
        with torch.no_grad():
            return self.model(board_features)

    def get_reward(self, reward, exp_return):
        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype, requires_grad=False) * reward


        exp_return = 0 # thessi lina laetur y[i] = reward * i
        # lata early moves fa expected return med sma nudge, late moves fa meira reward, a milli er progressive
        # y[seinast] = reward
        # y[0] er u.th.b. exp_return
        for i in range(episode_length):
            y[i] = (y[i] * i + (episode_length - (i + 1) ) * exp_return) / (episode_length - 1)

        loss = (self.predictions - y).pow(2).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.predictions - torch.ones((episode_length), dtype=dtype) * torch.mean(self.predictions))


        print("Prediction of last state ('-' means guessed wrong, number is confidence, optimal = 1 > p > 0.8) ")
        print(str(float(self.predictions[episode_length - 1] * reward)))
        print("First state")
        print(str(float(self.predictions[0] * reward)))
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()
