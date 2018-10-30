#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon


class RandomAgent(AgentInterface):

    def __init__(self):
        pass

    def action(self, board, dice, player):
        """
        Args:
            board (ndarray): backgammon board
            dice (ndarray): a pair of dice
            player: the number for the player on the board who's turn it is.

        Returns:
            A move `move`.
        """

        # check out the legal moves available for dice throw
        move = []
        possible_moves, _ = Backgammon.get_all_legal_moves_for_two_dice(board, dice)

        if len(possible_moves) == 0:
            return []
        else:
            move = possible_moves[np.random.randint(len(possible_moves))]
        
        return move
