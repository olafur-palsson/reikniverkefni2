#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: epsilon should be a parameter for this class, and also whether
one wants to use Parallel Network.
"""
import numpy as np
import random

from policy import Policy
from dyna2 import Dyna2
from pub_stomper_basic_network_for_testing import BasicNetworkForTesting
import torch

amount_of_planning_games = 10

class PolicyDyna2(Policy):

    # Epsilon for e-greedy
    # use agent_cfg['cfg']['epsilon'] instead
    # epsilon = 0.15

    # Decide what neural network to use
    # self.net = BasicNetworkForTesting()
    # or
    # self.net = ParallelNetwork() <-- little crazy
    def __init__(self, verbose=False, agent_cfg=None, imported=False):
        """
        Args:
            load_best (bool): default `False`
            verbose (bool): default `False`
            export (bool): default `False`
            agent_cfg: default `None`
            archive_name: default `None`.
        """
        if not agent_cfg:
            raise Error('yololo')

        Policy.__init__(self)
        self.verbose = verbose
        self.after_state = 'null'

        self.permanent_memory = torch.autograd.Variable(torch.ones((100, 1)))

        self.model = dict()
        self.plan = False

    # TODO
    # Make variables for the network
    # Make a second copy for the transient memory

    # At the start of each game do some planning
    #     This can be done by taking the permament memory,
    #     copy it
    #     then do the planning and update the shit based on the
    #     rewards and all that stuff

    # Next, take action with based on the transient memory
    # predict also with the permanent memory and update that based on
    # regular td-error

    # REMEMBER
    # Make it small
    # Use basic tools
    # Don't write super-dry code
    # Just get it done.

    def get_board_id(self, board):
        hex_numbers = []
        for feature in board:
            hex_numbers.append(hex(feature))
        return '-'.join(hex_numbers)

    def add_to_model(next_state):
        if current_state in model:
            model[self.after_state].append(next_state)
        else:
            model[self.after_state] = []
            model[self.after_state].append(next_state)

    def plan():
        print('do some planning here before we play the next game')

        transient_memory = BasicNetworkForTesting(agent_cfg=agent_cfg)


        self.should_plan = False

    def evaluate(self, possible_boards, board_copy):
        """
        Evaluates the possible boards given to this method as an argument and
        returns a move.

        Args:
            possible_boards: possible boards

        Returns:
            A move.
        """

        if self.should_plan:
            plan()
        # save the current state to the model only if not the starting state
        add_to_model(board_copy)

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

        self.after_state = get_board_id(possible_boards[move])

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
        self.after_state = 'null'
        # only necessary line in this function
        self.net.give_reward_to_nn(reward)
        self.should_plan = True
