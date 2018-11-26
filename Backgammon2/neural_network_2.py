#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_network_2.py
"""
import torch
import torch.nn as nn


# from torch.autograd import Variable


def make_layers(agent_cfg):
    layers_widths = []
    layers_widths += [ agent_cfg['cfg']['neural_network']['input_layer']   ]
    layers_widths +=   agent_cfg['cfg']['neural_network']['hidden_layers']
    layers_widths += [ agent_cfg['cfg']['neural_network']['output_layer']  ]
        # Total number of layers
    n = len(layers_widths)
    layers = []
    for i in range(n - 1):
        layer_width_left  = layers_widths[i]
        layer_width_right = layers_widths[i + 1]
        linear_module = nn.Linear(layer_width_left, layer_width_right)
        # layers.append(nn.ReLU()) # uncomment for ReLU
        # layers.append(nn.Dropout(p=0.025)) # uncomment for drop-out
        layers += [linear_module]
        # layers.append(nn.ReLU()) # uncomment for ReLU
    return layers


class NeuralNetwork2:

    def __init__(self, agent_cfg = None):

        torch.zeroes()

        layers = [
            nn.Linear(20, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])
    
    def forward(self, input):
        return output = self.model(input)