#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: epsilon should be a parameter for this class, and also whether
one wants to use Parallel Network.
"""
import numpy as np
import random

from pub_stomper_policy import Policy
from dyna2 import Dyna2
from backgammon_game import Backgammon
import torch

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
        self.verbose = verbose
        self.after_state = 'null'

        self.permanent_weights = torch.autograd.Variable(torch.ones((464, 1)))
        self.permanent_bias = torch.autograd.Variable(torch.ones((1, 1)))

        self.transient_weights = 'uninitialized'
        self.transient_bias = 'uninitialized'
        self.initialize_transient_memory()

        self.transient_predictions = []
        self.predictions = []

        self.model = dict()
        self.should_plan = False

    def transient_value_function(self, boards):
        features = list(map(lambda board: self.hot_one(board), boards))
        features = torch.stack(features)
        value = torch.mm(features, self.transient_weights) + self.transient_bias
        return value

    def permanent_value_function(self, boards):
        features = list(map(lambda board: self.hot_one(board), boards))
        features = torch.stack(features)
        values = torch.mm(features, self.permanent_weights) + self.permanent_bias
        return values

    def simulate_move(self, board, value_function='transient'):
        dice = Backgammon.roll_dice('yolo')
        possible_moves, possible_boards = Backgammon.get_all_legal_moves_for_two_dice(board, dice)

        if value_function == 'transient':
            move_values = self.transient_value_function(possible_boards)
        else:
            move_values = self.permanent_value_function(possible_board)

        move_values = move_values.squeeze()

        max = move_values[0]
        index_of_max = 0
        for i, value in enumerate(move_values):
            if value > max:
                index_of_max = 1
                max = value
        return max, index_of_max

    def winner(self, board):
        if board[28] > 14: return 1
        if board[29] > 14: return -1
        return 0

    def simulate_turn(self, board):
        move_value, best_move = simulate_move(board, value_function='transient')

        if winner(board) != 0:
            return possible_boards[best_move]

        self.transient_predictions.append(move_value)
        _, best_move = simulate_move(board, value_function='permanent')
        return possible_boards[best_move], best_move

    def simulate_remainder_of_episode(self, board):
        while not self.winner(board):
            board = simulate_turn(board)
        return winner(board)

    def initialize_transient_memory(self):
        self.transient_weights = self.permanent_weights.clone()
        self.transient_bias = self.permanent_bias.clone()

    def update_transient_weights(self, reward):
        copy = self.transient_predictions.stack().clone().detach()
        # not sure med ----->  [0:-1]
        target = torch.cat(copy[0:-1], reward * torch.ones((1, 1)))
        loss = (self.transient_predictions - target).pow(2).sum()
        self.transient_weights.zero_grad()
        self.transient_bias.zero_grad()
        loss.backward()

        self.transient_weights.data += learning_rate * self.transient_weights.grad.data
        self.transient_bias.data += learning_rate * self.transient_bias.grad.data
        self.transient_predictions = []

    def update_permanent_weights(self, reward):
        print(self.predictions)
        copy = torch.cat(self.predictions)
        print(copy.size())
        raise Error('Yolo')
        # not sure med ----->  [0:-1]
        target = torch.cat(copy[0:-1], reward * torch.ones((1, 1)))
        loss = (self.predictions - target).pow(2).sum()
        self.permanent_weights.zero_grad()
        self.permanent_bias.zero_grad()
        loss.backward()

        self.permanent_weights.data += learning_rate * self.permanent_weights.grad.data
        self.permanent_bias.data += learning_rate * self.permanent_bias.grad.data
        self.predictions = []

    def plan(self, board):
        print('do some planning here before we play the next game')
        self.should_plan = False
        initialize_transient_memory()
        for i in range(15):
            reward = simulate_remainder_of_episode(board)
            update_transient_weights(reward)

    def evaluate(self, possible_boards, board_copy):
        """
        Evaluates the possible boards given to this method as an argument and
        returns a move.

        Args:
            possible_boards: possible boards

        Returns:
            A move.
        """

        if len(possible_boards) == 1:
            return 0

        if self.should_plan:
            plan()

        move_value, move = self.simulate_move(board_copy)
        print(move_value)
        print(move)
        self.predictions.append(move_value)
        return move

    def save(self, save_as_best):
        print('Lol, nice try to SAVE the dyna2')

    def load(self, filename):
        print('Lol, nice try to LOAD the dyna2')

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
        self.update_permanent_weights(reward)
        self.should_plan = True
