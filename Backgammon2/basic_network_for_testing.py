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
import os.path

from lib.utils import load_file_as_json, get_random_string, rename_file_to_content_addressable, unarchive_archive, archive_files


dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

default_filename = "_".join(str(datetime.datetime.now()).split(" "))

default_agent_cfg = copy.deepcopy(load_file_as_json('configs/agent_nn_default.json'))



def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Linear') != -1:
        # print(m.weight)
        pass




class BasicNetworkForTesting():
    """
    Creates a basic neural network for testing.
    """



    # make this one output [nn.Linear, nn.Linear...] or whatever layers you would like, then the rest is automatic
    def make_layers(self, agent_cfg):
        """
        Create layers for neural network.
        David: Maybe this should be parameterized in the future?
        Oli:   It is possible but wouldn't that just move the setup
               data to a different location?
        David: Exactly.
        Oli:   Pretty clean bruh. •ᴗ•
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

    def make_filename_from_string(self, filename_root_string):
        # sets class-wide filename for exporting to files
        self.filename_model = filename_root_string + "_model.pt"
        self.filename_optimizer = filename_root_string + "_optim.pt"

    def parse_json(self, agent_cfg):
        self.cfg_sgd = agent_cfg['cfg']['sgd']
        self.cfg_neural_network = agent_cfg['cfg']['neural_network']
        self.name = agent_cfg['name']
        self.filename = agent_cfg['cfg']['filename']
        self.export = agent_cfg['cfg']['neural_network']['exported']

    def __init__(self, verbose=False, filename_of_network_to_bo_loaded = False, agent_cfg = None):
        """
        Args:
            filename_of_network_to_bo_loaded: default `False`
            export: default `False` 
            verbose: default `False`
            agent_cfg: default `False`
            archive_name: default `None`
        """
        self.verbose = verbose

        agent_cfg = agent_cfg if agent_cfg else default_agent_cfg
        self.parse_json(agent_cfg)

        # set up filenames for exporting
        self.make_filename_from_string(agent_cfg['name'])

        # If import if the config tells us to import it
        if self.cfg_neural_network['imported']:
            self.load(self.filename)
            return

        # make layers in neural network and make the network sequential
        # (i.e) input -> layer_1 -> ... -> layer_n -> output  for layers in 'make_layers()'
        self.model = nn.Sequential(*self.make_layers(agent_cfg))
        # loss function
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])

    def save(self, directory="./repository/"):
        """
        Exports everything related to the instantiation of this class to a
        ZIP file.

        Args:
            directory: directory where to place archive

        Returns:
            The path to the ZIP file.
        """
        print("SAVING...")

        # Save model
        filename_model = directory + self.filename_model
        torch.save(self.model.state_dict(), filename_model)
        # filename_model = rename_file_to_content_addressable(filename_model, ignore_extension=True, extension="_model.pt")

        # Save optimizer
        filename_optimizer = directory + self.filename_optimizer
        torch.save(self.optimizer.state_dict(), filename_optimizer)
        # filename_optimizer = rename_file_to_content_addressable(filename_optimizer, ignore_extension=True, extension="_optim.pt")

        """
        # Filenames
        filenames = [
            filename_settings,
            filename_model,
            filename_optimizer
        ]

        # Archive
        archive_name = directory + get_random_string(64)

        archive_files(archive_name, filenames, cleanup = True)
        archive_name = rename_file_to_content_addressable(archive_name, ignore_extension=True, extension="_bnft.zip")

        return archive_name
        """

    def load(self, archive_name):

        # CHECK IF FILE EXISTS
        if not os.path.isfile(self.filename_model) and not os.path.isfile(self.filename_optimizer):
            raise Exception('Did not find model or optimizer, export the model at least once first: \n',
                            '   Model name: ' + model_path,
                            '   Edit ./configs/agent_' + self.name + '.json so you have "imported: false" and "exported: true"')

        self.optimizer.load_state_dict(filename_optimizer)
        self.model.load_state_dict(filename_model)

    # initialize prediction storage
    predictions = torch.empty((1), dtype = dtype, requires_grad=True)
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

        if self.counter % 100 == 0 and self.export:
            self.export_model()

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
