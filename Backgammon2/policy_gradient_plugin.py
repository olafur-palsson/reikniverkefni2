
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

from lib.utils import load_file_as_json, get_random_string, rename_file_to_content_addressable, unarchive_archive, archive_files


dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU
default_filename = "_".join(str(datetime.datetime.now()).split(" "))


class PolicyGradientPlugin():
    # An add on for basic_network_for_testing.py (why is this name still there)

    def make_filename_from_string(self, filename_root_string):
        # sets class-wide filename for exporting to files
        self.filename_value_function = './repository/' + filename_root_string + "_valuefunction.pt"
        self.filename_value_optim  = './repository/' + filename_root_string + "_valueoptim.pt"
        self.filename_pg_model    = './repository/' + filename_root_string + "_pgmodel.pt"
        self.filename_pg_optim     = './repository/' + filename_root_string + "_pgoptim.pt"

    def parse_json(self, agent_cfg):
        self.cfg_sgd = agent_cfg['cfg']['sgd']
        self.cfg_neural_network = agent_cfg['cfg']['neural_network']
        self.name = agent_cfg['name']
        self.filename = agent_cfg['cfg']['filename']

    def __init__(self, verbose=False, agent_cfg = None, imported=False, use_sigmoid=False):

        """
        Args:
            filename_of_network_to_bo_loaded: default `False`
            export: default `False` 
            verbose: default `False`
            agent_cfg: default `False`
            archive_name: default `None`
        """
        agent_cfg = agent_cfg if agent_cfg else default_agent_cfg
        self.parse_json(agent_cfg)

        # set up filenames for exporting
        self.make_filename_from_string(self.filename)

        output_layer = self.cfg_neural_network['output_layer']

        self.softmax_function = nn.Softmax()

        # a.k.a. w2
        self.value_function = nn.Sequential(nn.Linear(output_layer, 1))
        self.value_optim = torch.optim.SGD(self.value_function.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])

        # a.k.a. theta
        self.pg_model = nn.Sequential(nn.Linear(output_layer, 100), nn.Linear(100, 1))
        self.pg_optim = torch.optim.SGD(self.pg_model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate_pg'])

        # If import if the config tells us to import it
        if imported:
            print('Loading...')
            self.load()
            return

    def save(self, save_as_best=False):
        if save_as_best:
            self.make_filename_from_string('nn_pg_best')

        print("Saving: " + self.filename_pg_model + ' and ' + self.filename_pg_optim)
        print("Saving: " + self.filename_value_function + ' and ' + self.filename_value_optim)

        # Save value function weights
        torch.save(self.value_function, self.filename_value_function)
        torch.save(self.value_optim.state_dict(), self.filename_value_optim)

        # Save policy gradient weights
        torch.save(self.pg_model, self.filename_pg_model)
        torch.save(self.pg_optim.state_dict(), self.filename_pg_optim)

    def load(self):
        print("Loading: " + self.filename_value_function + ' and ' + self.filename_value_optim)

        # CHECK IF FILE EXISTS
        filenames = [
            self.filename_value_function,
            self.filename_value_optim,
            self.filename_pg_model,
            self.filename_pg_optim
        ]
        if not all(filename for filename in filenames):
            raise Exception('Did not find all models or optimizers, export at least once first: \n',
                            '   Edit ./configs/agent_***.json so you have these: \n',
                            filenames)

        self.pg_model = torch.load(self.filename_pg_model)
        self.pg_optim = torch.optim.SGD(self.pg_model.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])
        self.pg_optim.load_state_dict(torch.load(self.filename_pg_optim))

        self.value_function = torch.load(self.filename_value_function)
        self.value_optim = torch.optim.SGD(self.value_function.parameters(), momentum = self.cfg_sgd['momentum'], lr = self.cfg_sgd['learning_rate'])
        self.value_optim.load_state_dict(torch.load(self.filename_value_optim))

    def softmax(self, input_vectors, requires_grad=True):
        scores = list(map(lambda input: self.pg_model(input.detach()), input_vectors))
        return Function.softmax(torch.cat(scores))

    def value_function(self, input, requires_grad=True):
        return self.value_function.apply(input)

    def manually_update_value_function(self):
        self.value_optim.step()

    def manually_update_pg_model(self):
        self.pg_optim.step()

    def reset_grads(self):
        self.pg_optim.zero_grad()
        self.value_optim.zero_grad()
