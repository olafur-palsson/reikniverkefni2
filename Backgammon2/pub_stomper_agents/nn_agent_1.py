#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A neural network agent.
"""
import numpy as np

from pub_stomper_agents.agent_interface import AgentInterface
from backgammon_game import Backgammon

from pub_stomper_policy_neural_network import PolicyNeuralNetwork
from pub_stomper_policy_pg_network import PolicyPGNetwork


class NNAgent1(AgentInterface):

    training = True

    def __init__(self, verbose=False, agent_cfg=None, imported=False):
        """
        Creates a neural network agent.

        To load the best NNAgent1 simply set load_best=True

        Args:
            load_best: default `False`
            verbose: default `False`
        """
        AgentInterface.__init__(self)
        if agent_cfg['cfg']['use_policy_gradient']:
            self.pub_stomper = PolicyPGNetwork(verbose=verbose, agent_cfg=agent_cfg, imported=imported)
        else:
            self.pub_stomper = PolicyNeuralNetwork(verbose=verbose, agent_cfg=agent_cfg, imported=imported)



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
            move = self.pub_stomper_policy(possible_moves, possible_boards, dice)

        return move

    def add_action(self, action):
        pass

    def export_model(self, filename=False):
        self.net.export_model(filename=filename)

    def add_reward(self, reward):
        """
        Adds reward `reward` to this neural network agent.

        NOTE: if you add a reward to the neural network it will immediately
        train.
        """

        # Hence, we only add rewards when we're training..
        if self.training:
            self.pub_stomper.add_reward(reward)

    def add_state(self, state):
        pass

    def load(self, filename):
        self.pub_stomper.load(filename)

    def save(self, save_as_best=False):
        return self.pub_stomper.save(save_as_best)

    def get_filename(self):
        # obsolete
        return self.pub_stomper.get_filename()

    def pub_stomper_policy(self, possible_moves, possible_boards, dice):

        best_move = self.pub_stomper.evaluate(possible_boards)
        move = possible_moves[best_move]

        # gamli kodinn fyrir random
        # move = possible_moves[np.random.randint(len(possible_moves))]
        return move
