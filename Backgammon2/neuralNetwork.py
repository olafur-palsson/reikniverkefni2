
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from torch.autograd import Variable

learning_rate = 0.01
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

input_width, output_width = 100, 1
hidden_layers_width = [1000, 100, 50, 50, 50, 50, 50, 50, 50, 55, output_width]

def make_layers():
    last_argument = input_width
    layers = []

    for width in hidden_layers_width:
        layer = torch.randn(last_argument, width, device=device, dtype=dtype)
        layer = Variable(layer, requires_grad = True)
        layers.append(layer)
        last_argument = width
    return layers

class NeuralNetwork(torch.autograd.Function):

    network = make_layers()
    predictions = []

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

    def run_decision(self, board_features, abs_of_max = 1):
        vector = board_features
        vector = self.apply(torch.mv(self.network[0].t(), board_features))
        skipped_first = False
        for layer in self.network:
            if not skipped_first:
                skipped_first = True
                continue
            vector = torch.mv(layer.t(), vector)
        y_pred = vector / abs_of_max
        print(torch.log(y_pred))
        self.predictions.append(y_pred)

    def evaluate(self, board_features):
        with torch.no_grad():
            current_vector = board_features
            for layer in self.network:
                current_vector = self.apply(torch.mv(layer.t(), current_vector))
            return current_vector

    def get_reward(self, reward):
        episode_length = len(self.predictions)
        tensor  = torch.ones(1, dtype=dtype)
        y = tensor.new_full((1, 1), reward, dtype=dtype)
        losses = []

        for prediction in self.predictions:
            loss = ((y - prediction).pow(2).sum() / episode_length)
            loss.backward()
            losses.append(loss)

        self.predictions = []
        map(lambda loss : loss.backward, losses)
        with torch.no_grad():
            layers = map(lambda layer : layer - learning_rate * layer.grad, self.network)
            map(lambda layer : layer.zero_grad, self.network)
