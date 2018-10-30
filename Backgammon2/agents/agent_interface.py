#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

class AgentInterface():

    def __init__(self):
        pass

    def action(self, board, dice, player):
        """
        The action returns a list of two list, and each of those sub-lists
        contain two numbers, e.g.

        [ [18, 16], [16, 10] ]

        We move a checker from 18 to 16 and a checker from 16 to 10.

        Args:
            board (ndarray): backgammon board
            dice (ndarray): a pair of dice
            player: the number for the player on the board who's turn it is.

        Returns:
            A move `move`.
        """
        raise Exception("Not implemented!")
