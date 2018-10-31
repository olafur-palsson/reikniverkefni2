#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A neural network agent.
"""
import numpy as np

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon

from policy_neural_network import PolicyNeuralNetwork


class NNAgent1(AgentInterface):

    def __init__(self, load_best=False, verbose=False):

        self.pub_stomper = PolicyNeuralNetwork(verbose=verbose)

    def action(self, board, dice, player):
        """
        Args:
            board (ndarray): backgammon board
            dice (ndarray): a pair of dice
            player: the number for the player on the board who's turn it is.

        Returns:
            A move `move`.
        """

        move = []
        possible_moves, possible_boards = Backgammon.get_all_legal_moves_for_two_dice(board, dice)

        if len(possible_moves) != 0:
            move = self.policy(possible_moves, possible_boards, dice)

        return move


    def reward_player(self, reward):
        self.pub_stomper.add_reward(reward)


    def get_file_name(self):
        return self.pub_stomper.get_file_name()


    def policy(self, possible_moves, possible_boards, dice):

        best_move = self.pub_stomper.evaluate(possible_boards)
        move = possible_moves[best_move]

        # gamli kodinn fyrir random
        # move = possible_moves[np.random.randint(len(possible_moves))]
        return move
