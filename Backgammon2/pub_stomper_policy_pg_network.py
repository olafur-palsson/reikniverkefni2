
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: epsilon should be a parameter for this class, and also whether
one wants to use Parallel Network.
"""
# Basic libraries
import numpy as np
import random
import torch

from torch.distributions.categorical import Categorical
from torch.distributions.multinomial import Multinomial

from pub_stomper_policy import Policy
from pub_stomper_basic_network_for_testing import BasicNetworkForTesting
from pub_stomper_policy_gradient_plugin import PolicyGradientPlugin

class PolicyPGNetwork(Policy):

    def __init__(self, verbose=False, agent_cfg=None, imported=False, pub_stomper_policy_decision_function='argmax'):
        """
        Args:
            load_best (bool): default `False`
            verbose (bool): default `False`
            export (bool): default `False`
            agent_cfg: default `None`
            archive_name: default `None`.
        """
        if not agent_cfg:
            print('No cfg file bruh')
        Policy.__init__(self)
        self.pub_stomper_policy_decision_function = pub_stomper_policy_decision_function
        self.saved_log_probabilities = []
        self.saved_value_estimations = []

        self.verbose = verbose
        self.net = BasicNetworkForTesting(verbose=verbose, agent_cfg=agent_cfg, imported=imported)
        self.pg_plugin = PolicyGradientPlugin(agent_cfg=agent_cfg, imported=imported)

    def argmax(self, move_ratings):
        # get max value
        max = move_ratings[0]
        max_i = 0
        for i, move in enumerate(move_ratings):
            if move > max:
                max = move
                max_i = i
        return max_i

    # initialize move storage
    # YOU ARE NEVER EMPTYING THIS STUFF

    def run_through_neural_network(self, possible_boards):
        last_layer_outputs = []
        for board in possible_boards:
            value_of_board = self.net.run_decision(self.get_feature_vector(board), save_predictions=False)
            layst_layer_outputs = last_layer_outputs.append(value_of_board)
        return last_layer_outputs


    def evaluate(self, possible_boards):

        # possible_boards -> neural network -> sigmoid -> last_layer_sigmoid
        last_layer_outputs = self.run_through_neural_network(possible_boards)
        # last_layer_sigmoid = list(map(lambda x: x.sigmoid(), last_layer_outputs))

        # Decide move and save log_prob for backward
        # We make sure not to affect the value fn with .detach()

        probs = self.pg_plugin.softmax(last_layer_outputs)
        distribution = Multinomial(1, probs)
        move = distribution.sample()
        self.saved_log_probabilities.append(distribution.log_prob(move))

        _, move = move.max(0)
        # calculate the value estimation and save for backward
        value_estimate = self.pg_plugin.value_function(last_layer_outputs[move])
        self.saved_value_estimations.append(value_estimate)
        return move

    def save(self, save_as_best=False):
        self.net.save(save_as_best=save_as_best)
        self.pg_plugin.save(save_as_best=save_as_best)

    def load(self, filename):
        self.net.load(filename)
        self.pg_plugin.load(filename)

    def get_filename(self):
        """
        Returns the file name for this neural network attached to this instance.

        Returns:
            The file name of the neural network.
        """
        return self.net.filename

    def add_reward(self, reward):
        td_n = 1
        episode_length = len(self.saved_value_estimations)
        values = torch.stack(self.saved_value_estimations).squeeze()
        values_no_grad = values.detach()
        # shift the values by td_n
        targets = torch.cat((values_no_grad[td_n:], (torch.ones(td_n) * reward)))
        # Squared error for value function
        loss = (targets - values).pow(2).sum()

        self.pg_plugin.reset_grads()
        self.net.manually_reset_grad()

        # Update the weights of value function
        loss.backward()
        self.net.manually_update_weights_of_network()
        self.pg_plugin.manually_update_value_function()

        # til ad optimize-a thetta tha tharf eg ad setja thetta i module
        # like this guy
        # https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py#L90
        # Update the pub_stomper_policy gradient by maximizing this:
        rewards = torch.ones(episode_length) * reward
        # get the sum of the rewards * log_prob a.k.a. loss
        # Note that we put the (-) in front of rewards to do
        # gradient ascent instead of descent
        log_probs = torch.stack(self.saved_log_probabilities)
        loss2 = torch.dot(-rewards, log_probs)
        loss2.backward()
        self.pg_plugin.manually_update_pg_model()

        self.saved_log_probabilities = []
        self.saved_value_estimations = []
