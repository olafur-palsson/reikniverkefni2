#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



TODO: epsilon should be a parameter for this class, and also whether
one wants to use Parallel Network.
"""
import numpy as np
import random

from policy import Policy
from basic_network_for_testing import BasicNetworkForTesting
from parallel_network import ParallelNetwork




class PolicyNeuralNetwork(Policy):

    # Epsilon for e-greedy
    # use agent_cfg['cfg']['epsilon'] instead
    # epsilon = 0.15

    # Data for statistics
    number_of_decisions_0 = 0
    decision_counter = 0
    counter = 0
    net = 0

    # Decide what neural network to use
    # self.net = BasicNetworkForTesting()
    # or
    # self.net = ParallelNetwork() <-- little crazy
    def __init__(self, load_best = False, verbose = False, export = False, agent_cfg = None, archive_name = None):
        """


        Args:
            load_best (bool): default `False`
            verbose (bool): default `False`
            export (bool): default `False`
            agent_cfg: default `None`
            archive_name: default `None`.
        """

        Policy.__init__(self)

        self.verbose = verbose

        self.net = BasicNetworkForTesting(verbose = verbose, export = export, agent_cfg = agent_cfg, archive_name=archive_name)

        if False:
            if load_best:
                self.net = BasicNetworkForTesting(file_name_of_network_to_bo_loaded = "nn_best", verbose = verbose, export = True, agent_cfg = agent_cfg)
            else:
                self.net = BasicNetworkForTesting(verbose = verbose, export = export, agent_cfg = agent_cfg)


    def evaluate(self, possible_boards):
        """
        Evaluates the possible boards given to this method as an argument and
        returns a move.

        Args:
            possible_boards: possible boards

        Returns:
            A move.
        """

        # variable to hold ratings
        move_ratings = []

        # predict win_rate of each possible after-state (possible_boards)
        for board in possible_boards:
            value_of_board = self.net.predict(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        # get max value
        max = move_ratings[0]
        max_i = 0
        for i, move in enumerate(move_ratings):
            if move > max:
                max = move
                max_i = i


        best_move = max_i
        move = best_move
        self.number_of_decisions_0 += int(move == 0)
        self.decision_counter += 1
        # move = best_move if random.random() > self.epsilon else random.rand_int(len(possible_boards - 1)) # uncomment for e_greedy
        self.net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move

    def save(self):
        return self.net.save()
    
    def load(self, filename):
        self.net.load(filename)

    def get_file_name(self):
        """
        Returns the file name for this neural network attached to this instance.

        Returns:
            The file name of the neural network.
        """
        return self.net.file_name


    def log_and_reset_number_of_decisions_0(self):
        if self.verbose:
            print("")
            print("% of decisions '0' (first of array), lower is better ")
            print(str(float(self.number_of_decisions_0) / self.decision_counter))
        self.number_of_decisions_0 = 0
        self.decision_counter = 0


    def export_network(self, file_name=False):
        self.net.export(file_name=file_name)


    def add_reward(self, reward):
        # only necessary line in this function
        self.net.give_reward_to_nn(reward)

        # statistics
        self.counter += 1

        self.log_and_reset_number_of_decisions_0()
