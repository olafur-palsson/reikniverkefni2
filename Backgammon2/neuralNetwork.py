

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

input_width, output_width = 100, 1

hidden_layers_width = [50, 50 ,50, 50, 50, 50, output_width]

def make_layers():
    last_argument = input_width
    layers = []

    for width in hidden_layers_width:
        layer = torch.randn(last_argument, width)
        layers.append(layer)
        last_argument = width

    return layers

class NeuralNetwork(torch.autograd.Function):

    network = make_layers()

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

    def decide(board):
        y_pred = reduce((lambda x, layer : self.apply(x.mm(layer))), network)

    def evaluate(board):
        with torch.no_grad():
            return reduce((lambda x, layer : self.apply(x.mm(layer))), network)

    # Eg veit ekki alveg hvernig vid eigum ad gera thetta
    # Held eg se a rangri leid

    # Oll guides a netinu er med y_pred sem utkomu og nota hana til ad
    # gera backprop, vid erum hins vegar med mikid af actions en eina utkomu

    # Hvernig uppfaerum vid value fyrir allar stodurnar inn a milli?
    # Thurfum vid ad update-a value fyrir allar stodur inn a milli yfirhofud?
    # Hvernig litur grafid ut af update-inu?
    # Eru morg output kannski?
    # Hvad er eg ekki ad sja?

    # Basically: Hvernig (oft forward i nn) -> reward -> (backprop update)
    def get_reward(reward):
        y = np.array([reward])
        einhver_med_hugmyndir = 0
        return einhver_med_hugmyndir






