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
import os

from pub_stomper_lib.utils import load_file_as_json, get_random_string, rename_file_to_content_addressable, unarchive_archive, archive_files


dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

default_filename = "_".join(str(datetime.datetime.now()).split(" "))

default_agent_cfg = copy.deepcopy(load_file_as_json('pub_stomper_configs/agent_nn_default.json'))

class BasicNetworkForTesting():
    """
    Creates a basic neural network for testing.
    """

    # make this one output [nn.Linear, nn.Linear...] or whatever layers you would like, then the rest is automatic
    def make_layers(self, agent_cfg):
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
        try:
            if self.cfg_neural_network['sigmoid']:
                layers += [torch.nn.Sigmoid()]
        except KeyError:
            do_nothing = 0
        return layers

    def make_filename_from_string(self, filename_root_string):
        # sets class-wide filename for exporting to files
        self.filename_model = './pub_stomper_repository/' + filename_root_string + "_model.pt"
        self.filename_optimizer = './pub_stomper_repository/' + filename_root_string + "_optim.pt"

    def parse_json(self, agent_cfg):
        self.cfg_sgd = agent_cfg['cfg']['sgd']
        self.cfg_neural_network = agent_cfg['cfg']['neural_network']
        self.name = agent_cfg['name']
        self.filename = agent_cfg['cfg']['filename']

    def __init__(self, verbose=False, filename_of_network_to_bo_loaded = False, agent_cfg = None, imported=False, use_sigmoid=False):

        """
        Args:
            filename_of_network_to_bo_loaded: default `False`
            export: default `False` 
            verbose: default `False`
            agent_cfg: default `False`
            archive_name: default `None`
        """
        self.verbose = verbose
        self.use_sigmoid = use_sigmoid
        agent_cfg = agent_cfg if agent_cfg else default_agent_cfg
        self.parse_json(agent_cfg)

        # set up filenames for exporting
        self.make_filename_from_string(self.filename)

        self.model = nn.Sequential(*self.make_layers(agent_cfg))
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])

        # If import if the config tells us to import it
        if imported:
            self.load()
            return

        # make layers in neural network and make the network sequential
        # (i.e) input -> layer_1 -> ... -> layer_n -> output  for layers in 'make_layers()'
        # loss function
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)
        # optimizer

    def save_clone(self, name):
        torch.save(self.model, 'clone_model_' + name)
        torch.save(self.optimizer.state_dict(), 'clone_optim_' + name)

    def load_clone(self, name):
        self.model = torch.load('clone_model_' + name)
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])
        self.optimizer.load_state_dict(torch.load('clone_optim_' + name))

    def save(self, save_as_best=False):
        """
        Exports everything related to the instantiation of this class to a
        ZIP file.
        Args:
            directory: directory where to place archive

        Returns:
            The path to the ZIP file.
        """
        if save_as_best:
            self.make_filename_from_string('nn_best')

        print("Saving: " + self.filename_model + ' and ' + self.filename_optimizer)

        # Save model
        torch.save(self.model, self.filename_model)
        # filename_model = rename_file_to_content_addressable(filename_model, ignore_extension=True, extension="_model.pt")

        # Save optimizer
        torch.save(self.optimizer.state_dict(), self.filename_optimizer)
        # filename_optimizer = rename_file_to_content_addressable(filename_optimizer, ignore_extension=True, extension="_optim.pt")

        self.make_filename_from_string(self.filename)

        return self.filename_model, self.filename_optimizer


    def load(self):
        print("Loading: " + self.filename_model + ' and ' + self.filename_optimizer)
        # CHECK IF FILE EXISTS
        if not os.path.isfile(self.filename_model) and not os.path.isfile(self.filename_optimizer):
            raise Exception('Did not find model or optimizer, export the model at least once first: \n',
                            '   Model name: ' + self.filename_model,
                            '   Edit ./pub_stomper_configs/agent_' + self.name + '.json so you have "imported: false" and "exported: true"')

        self.model = torch.load(self.filename_model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])
        self.optimizer.load_state_dict(torch.load(self.filename_optimizer))


    # initialize prediction storage
    predictions = torch.empty((1), dtype = dtype, requires_grad=True)

    # run a feature vector through the model accumulating greadient
    def run_decision(self, board_features, save_predictions=True):
        prediction = self.model(board_features)
        if save_predictions:
            self.predictions = torch.cat((self.predictions, prediction.double()))
        return prediction

    # run a feature vector through the model without accumulating gradient
    def predict(self, board_features):
        """
        This method behaves like the value function.

        Note:
        'with torch.no_grad()' allows us to run things through the network without
        calculating gradients. Although it doesn't affect the learning-process of the
        network it does save a lot of computing power.

        Args:
            board_features (ndarray or list): the feature vector for the board under consideration.

        Returns:
            The value of the board represented by the feature vector
        """
        with torch.no_grad():
            # This inputs the feature vector (features) into the neural
            # network, denoted `self.model` and outputs a number (the value of
            # the board).
            return self.model(board_features)


    def manually_reset_grad(self):
        self.optimizer.zero_grad()
    # for use in pub_stomper_policy gradient
    def manually_update_weights_of_network(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Initialize reward storage and statistical variables
    rewards = []
    counter = 0
    last_500_wins = np.zeros(500)
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
                if i == len(self.predictions) - self.cfg_neural_network['temporal_delay']:
                    break
                y[i] = self.predictions[i + self.cfg_neural_network['temporal_delay']]

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

        if self.verbose:
            # Log out statistics of current game
            self.last_500_wins[self.counter % 500] = reward
            exp_return = np.sum(self.last_500_wins) / 500 # this is from -1 to 1
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
