
import torch
import torch.nn as nn
import itertools
from functools import reduce

input_width = 464
output_width = 50
hidden_layers = [100,50,100]
dtype=torch.double

learning_rate = 5e-5
node_count = 3
last_vector = node_count * output_width

def make_layers():
    layers = []

    last_width = input_width
    for layer_width in hidden_layers:
        layers.append(nn.Linear(last_width, layer_width))
        layers.append(nn.ReLU())
        last_width = layer_width
    final = nn.Linear(last_width, output_width)

    layers.append(final)
    return layers

class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        self.model = nn.Sequential(*make_layers())

    def forward(self, board):
        return self.model(board)

    def predict(self, board):
        with torch.no_grad():
            return self.model(board)

    def run_decision(self, board):
        return self.model(board)

def make_nodes(n):

    nodes = []
    for i in range(n):
        nodes.append(Node())
    return nodes

def get_parameters(nodes):
    chained = []
    for node in nodes:
        itertools.chain(chained, node.parameters())
    return chained

class ParallelNetwork(nn.Module):

    def __init__(self):
        super(ParallelNetwork, self).__init__()
        self.predictions = torch.empty((1), dtype = dtype, requires_grad=True)
        self.nodes = make_nodes(node_count)
        self.prefinal = nn.Linear(last_vector, 100)
        self.final = nn.Linear(100, 1)
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(itertools.chain(self.prefinal.parameters(), self.final.parameters(), get_parameters(self.nodes)), lr=learning_rate)

    def forward(self, board):
        tensor = []
        for node in self.nodes:
            tensor.append(node(board))
        node_out = torch.stack(tensor)
        node_out = node_out.view((last_vector))
        digestion = self.prefinal(node_out)
        output = self.final(digestion)
        return output

    def run_decision(self, board_features):
        vector = board_features
        prediction = self(board_features)
        self.predictions = torch.cat((self.predictions, prediction.double()))

    def predict(self, board_features):
        with torch.no_grad():
            return self(board_features)

    def get_reward(self, reward, exp_return):
        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype) * reward

        for i in range(episode_length):
            y[i] = (y[i] * i + (episode_length - (i + 1) ) * exp_return) / (episode_length - 1)


        loss = (self.predictions - y).pow(2).sum() / episode_length
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.predictions - torch.ones((episode_length), dtype=dtype) * torch.mean(self.predictions))

        print("Expected return")
        print(exp_return)
        print("")
        print("Prediction of last state ('-' means guessed wrong, number is confidence, optimal = 1 > p > 0.8) ")
        print(str(float(self.predictions[episode_length - 1] * y[0])))
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()
