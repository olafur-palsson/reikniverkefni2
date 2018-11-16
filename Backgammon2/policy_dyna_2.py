#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: epsilon should be a parameter for this class, and also whether
one wants to use Parallel Network.
"""
import numpy as np
import random

from policy import Policy

class PolicyDyna2(Policy):

    # Epsilon for e-greedy
    # use agent_cfg['cfg']['epsilon'] instead
    # epsilon = 0.15

    # Data for statistics
    counter = 0
    net = 0

    # Decide what neural network to use
    # self.net = BasicNetworkForTesting()
    # or
    # self.net = ParallelNetwork() <-- little crazy
    def __init__(self, verbose = False, agent_cfg = None, imported=False):
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
        self.dyna2 = Dyna2()

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
            value_of_board = self.dyna2.predict(self.get_feature_vector(board))
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

        # self.net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move

    def save(self, save_as_best):
        return self.dyna2.save()

    def load(self, filename):
        self.dyna2.load(filename)

    def get_filename(self):
        """
        Returns the file name for this neural network attached to this instance.

        Returns:
            The file name of the neural network.
        """
        return self.net.filename

    def add_reward(self, reward):
        # only necessary line in this function
        self.net.give_reward_to_nn(reward)
        # statistics
        self.counter += 1
