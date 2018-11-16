
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A neural network agent.
"""
import numpy as np

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon

from policy_dyna_2 import Dyna2


class Dyna2Agent(AgentInterface):

    training = True

    def __init__(self, verbose=False):
        self.dyna2_policy = PolicyDyna()
        """
        Creates a neural network agent.

        To load the best NNAgent1 simply set load_best=True

        Args:
            load_best: default `False`
            verbose: default `False`
        """
        AgentInterface.__init__(self)


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

    def add_action(self, action):
        pass

    def add_reward(self, reward):
        """
        Adds reward `reward` to this neural network agent.

        NOTE: if you add a reward to the neural network it will immediately
        train.
        """
        # Hence, we only add rewards when we're training..
        raise Exception('Add reward not implemented')

    def add_state(self, state):
        raise Exception('Add')
        pass

    def load(self, filename):
        self.pub_stomper.load(filename)

    def save(self, save_as_best=False):
        return self.pub_stomper.save(save_as_best)

    def get_filename(self):
        # obsolete
        return self.pub_stomper.get_filename()

    def policy(self, possible_moves, possible_boards, dice):
        best_move = self.dyna2.evaluate()
        move = possible_moves[best_move]

        # gamli kodinn fyrir random
        # move = possible_moves[np.random.randint(len(possible_moves))]
        return move
