#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: epsilon should be a parameter for this class, and also whether
one wants to use Parallel Network.
"""
# Basic libraries
import numpy as np
import random
import time
import copy
import torch

# Recycled code
from pub_stomper_policy import Policy
from backgammon_game import Backgammon
from pub_stomper_basic_network_for_testing import BasicNetworkForTesting

# Dyna2 specific code
from psuedo_policy import PolicyPsuedo
from pub_stomper_agents.psuedo_agent import PsuedoAgent

amount_of_planning_games = 10
learning_rate = 0.005

class PolicyDyna2(Policy):

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
            print('No cfg file bruh')

        Policy.__init__(self)
        self.agent_cfg = agent_cfg
        self.verbose = verbose
        self.after_state = 'null'
        self.permanent_net = BasicNetworkForTesting(verbose=verbose, agent_cfg=agent_cfg, imported=imported)
        self.transient_net = 'uninitialized'
        self.model_net = 'uninitialized'

    def winner(self, board):
        if board[28] > 14: return 1
        if board[29] > 14: return -1
        return 0

    def simulate_remainder_of_episode(self, board):
        while not self.winner(board):
            board = self.simulate_turn(board)
        return winner(board)

    def clone_neural_network(self, net, name):
        net.save_clone(name)
        clone = BasicNetworkForTesting(agent_cfg=self.agent_cfg)
        clone.load_clone(name)
        return clone

    def initialize_planning_phase(self):
        self.transient_net = self.clone_neural_network(self.permanent_net, 'kloni_doni')
        self.model_net = self.clone_neural_network(self.permanent_net, 'model')
        model_agent = PsuedoAgent(PolicyPsuedo(self.model_net))
        transient_agent = PsuedoAgent(PolicyPsuedo(self.transient_net))
        return model_agent, transient_agent

    def plan(self, board):
        time_when_planning_should_stop = time.time() * 1000 + 250
        model_agent, transient_agent = self.initialize_planning_phase()
        while time.time() * 1000 < time_when_planning_should_stop:
            copy_of_board = copy.deepcopy(board)
            game = Backgammon()
            game.reset()
            game.set_player_1(model_agent)
            game.set_player_2(transient_agent)
            reward = game.play(start_with_this_board=copy_of_board)
            transient_agent.add_reward(reward)

    def evaluate(self, possible_boards, board_copy):
        # variable to hold ratings
        move_ratings = []

        self.plan(board_copy)

        # predict win_rate of each possible after-state (possible_boards)
        for board in possible_boards:
            value_of_board = self.transient_net.predict(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        move = 0
        best_rating = move_ratings[0]
        for i, rating in enumerate(move_ratings):
            if best_rating < rating:
                best_rating = rating
                move = i

        # move = best_move if random.random() > self.epsilon else random.rand_int(len(possible_boards - 1)) # uncomment for e_greedy
        self.permanent_net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move


    def save(self, save_as_best):
        return self.permanent_net.save(save_as_best=save_as_best)

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
        self.permanent_net.give_reward_to_nn(reward)
