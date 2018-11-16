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
    policy = 'argmax'

    # Decide what neural network to use
    # self.net = BasicNetworkForTesting()
    # or
    # self.net = ParallelNetwork() <-- little crazy
    def __init__(self, verbose = False, agent_cfg = None, imported=False, policy_decision_function='argmax'):
        """
        Args:
            load_best (bool): default `False`
            verbose (bool): default `False`
            export (bool): default `False`
            agent_cfg: default `None`
            archive_name: default `None`.
        """
        self.policy_decision_function = policy_decision_function

        Policy.__init__(self)
        self.verbose = verbose
        self.net = BasicNetworkForTesting(verbose = verbose, agent_cfg = agent_cfg, imported=imported)

    def argmax(self, move_ratings):
        # get max value
        max = move_ratings[0]
        max_i = 0
        for i, move in enumerate(move_ratings):
            if move > max:
                max = move
                max_i = i
        return max_i

    def policy_gradient(self, move_ratings):
        exponential_ratings = map(lambda move_rating: np.e ** move_rating)
        move = 0
        random_number = random.random()
        accumulator = 0
        for rating in exponential_ratings:
            accumulator += rating
            if accumulator > random_number:
                break
            move += 1

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

        move = 0
        self.number_of_decisions_0 += int(move == 0)
        self.decision_counter += 1
        # move = best_move if random.random() > self.epsilon else random.rand_int(len(possible_boards - 1)) # uncomment for e_greedy
        self.net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move

    def save(self, save_as_best=False):
        return self.net.save(save_as_best=save_as_best)

    def load(self, filename):
        self.net.load(filename)

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
