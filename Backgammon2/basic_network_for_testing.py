#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This *IS* the neural network under consideration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as Optimizer
from torch.autograd import Variable
import numpy as np
import datetime
from functools import reduce
from pathlib import Path
import copy

from lib.utils import load_file_as_json


dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

default_file_name = "_".join(str(datetime.datetime.now()).split(" "))

default_agent_cfg = load_file_as_json('configs/agent_nn_default.json')


# make this one output [nn.Linear, nn.Linear...] or whatever layers you would like, then the rest is automatic
def make_layers(agent_cfg):
    """
    Create layers for neural network.

    David: Maybe this should be parameterized in the future?
    Oli:   It is possible but wouldn't that just move the setup
           data to a different location?
    David: Exactly.
    """

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







class BasicNetworkForTesting():
    """
    Creates a basic neural network for testing.
    """

    def __init__(self, file_name_of_network_to_bo_loaded = False, export = False, verbose = False, agent_cfg = None):
        """
        Previous parameters: (self, )

        Args:
            file_name_of_network_to_bo_loaded: default `False`

        """

        if agent_cfg is None:
            agent_cfg = copy.deepcopy(default_agent_cfg)
        
        self.agent_cfg = agent_cfg

        file_name_of_network_to_bo_loaded = file_name_of_network_to_bo_loaded
        export = export
        verbose = verbose
        
        self.agent_cfg = agent_cfg

        self.last_500 = np.zeros(500)
        self.verbose = verbose
        # set up file_names for exporting
        self.file_name = file_name_of_network_to_bo_loaded if file_name_of_network_to_bo_loaded else default_file_name

        if True:
            print("TEST:::", self.file_name)

        self.make_file_name_from_string(self.file_name)

        # make layers in neural network and make the network sequential
        # (i.e) input -> layer_1 -> ... -> layer_n -> output  for layers in 'make_layers()'
        self.model = nn.Sequential(*make_layers(self.agent_cfg))

        # initialize prediction storage
        self.predictions = torch.empty((1), dtype = dtype, requires_grad=True)

        # set loss function for backprop (usage is optional)
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)

        # set optimizer for adjusting the weights (e.g Stochastic Gradient Descent, SGD)
        # Note: learning_rate is at the top of the script
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum = self.agent_cfg['cfg']['sgd']['momentum'], lr = self.agent_cfg['cfg']['sgd']['learning_rate'])

        # True if should export, False if we throw it away after running
        self.export = export

        # Game counter for intervals
        self.counter = 0

        # Reward storage for batched learning
        self.rewards = []

        # If we want to load a model we input the name of the file, if exists -> load
        if file_name_of_network_to_bo_loaded:
            # import model
            self.optimizer.load_state_dict(torch.load("./exported_networks/" + self.file_name + "_optim.pt"))
            self.model.load_state_dict(torch.load("./exported_networks/" + self.file_name + "_model.pt"))
        else:
            # export current settings
            self.make_settings_file()

    def make_settings_file(self):
        Path(self.settings_file_name).touch()
        file = open(self.settings_file_name, "w")
        file.write("Input vector size: " + str(self.agent_cfg['cfg']['neural_network']['input_layer']) + "\n")
        file.write("Hidden layers: " + str(self.agent_cfg['cfg']['neural_network']['hidden_layers']) + "\n")
        file.write("Learning rate: " + str(self.agent_cfg['cfg']['sgd']['learning_rate']) + "\n")
        file.close()


    def make_file_name_from_string(self, file_name_root_string):
        # sets class-wide filename for exporting to files
        self.model_file_name = "./exported_networks/" + file_name_root_string + "_model.pt"
        self.optimizer_file_name = "./exported_networks/" + file_name_root_string + "_optim.pt"
        self.settings_file_name = "results/" + file_name_root_string + "_settings.pt"

    def export_model(self, file_name=False):
        if file_name:
            torch.save(self.model.state_dict(), "./exported_networks/" + file_name + "_model.pt")
            torch.save(self.optimizer.state_dict(), "./exported_networks/" + file_name + "_optim.pt")
        else:
            torch.save(self.model.state_dict(), self.model_file_name)
            torch.save(self.optimizer.state_dict(), self.optimizer_file_name)


    # run a feature vector through the model accumulating greadient
    def run_decision(self, board_features):
        prediction = self.model(board_features)
        self.predictions = torch.cat((self.predictions, prediction.double()))


    # run a feature vector through the model without accumulating gradient
    def predict(self, board_features):
        """
        Returns the value of the board represented by the feature vector
        `board_features`.

        This method behaves like the value function.

        Note:
        'with torch.no_grad()' allows us to run things through the network without
        calculating gradients. Although it doesn't affect the learning-process of the
        network it does save a lot of computing power.

        Args:
            board_features (ndarray or list): the feature vector for the board under consideration.

        Returns:
            The value of the board
        """
        with torch.no_grad():

            # This inputs the feature vector (features) into the neural
            # network, denoted `self.model` and outputs a number (the value of
            # the board).
            return self.model(board_features)


    # Function run on the end of each game.
    def give_reward_to_nn(self, reward):
        """

        TODO: this is problematic because we might not want to train the network yet,
        i.e. maybe we want to accumulate rewards and games then train

        We at this point have accumulated predictions of the network in self.predictions
        Here we decide what values we should move towards. We shall name that
        vector 'y'

        NOTE: This method does all the learning.


        Args:
            reward (number): the reward (a scalar)
            verbose (boolean): print out log for details
        """

        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype, requires_grad=False) * reward

        # TD valued reward
        with torch.no_grad():
            for i in range(len(self.predictions)):
                if i == len(self.predictions) - self.agent_cfg['cfg']['temporal_delay']:
                    break
                y[i] = self.predictions[i + self.agent_cfg['cfg']['temporal_delay']]

        self.rewards.append(y)

        # Sum of squared error as loss
        loss = (self.predictions - y).pow(2).sum()
        # Zero all accumulated gradients
        self.optimizer.zero_grad()
        # Recalculate gradients based on 'loss' (i.e. what it takes for loss -> 0)
        loss.backward()
        # Use optimizer to calculate new weights
        self.optimizer.step()

        # Export model each 100 episodes
        self.counter += 1
        if self.counter % 100 == 0 and self.export:
            self.export_model()

        if self.verbose:
            # Log out statistics of current game
            self.last_500[self.counter % 500] = reward
            exp_return = np.sum(self.last_500) / 500 # this is from -1 to 1
            print(self.counter)
            print("")
            print("Expected return")
            print(exp_return)
            print("First state td value")
            print(y[0])
            print("Prediction of last state ('-' means guessed wrong, number is confidence, optimal = 1 > p > 0.8) ")
            print(str(float(self.predictions[episode_length - 1] * reward)))
            print("First state")
            print(str(float(self.predictions[0])))

        # reset empty predictions
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()
