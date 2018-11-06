#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class, or interface, for agents. 
"""

class AgentInterface():
    """
    A standardized interface for agents.
    """

    def __init__(self, training = False):
        """
        Instantiates an standardized agent interface.

        Args:
            training (bool): whether this agent is training, default `False`
        """

        self.training = training


    def action(self, board, dice, player):
        """
        This method returns a list of two least, and ea contain two numbers, 
        e.g.

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


    def add_action(self, action):
        """
        Adds action `action`.

        Args:
            action: the action.
        """
        raise Exception("Not implemented!")


    def add_reward(self, reward):
        """
        Adds reward `reward`.

        Args:
            reward (number): the reward
        """
        raise Exception("Not implemented!")
    

    def add_state(self, state):
        """
        Adds state `state`.

        Args:
            state: the state
        """
        raise Exception("Not implemented!")


    def load(self, filepath = None):
        """
        Loads agent from disk.

        NOTE: Refrain from using `filepath`.

        Returns:
            Path to where the file is saved.
        """
        raise Exception("Not implemented!")
    

    def save(self, filepath = None):
        """
        Saves agent to disk.

        NOTE: Refrain from using `filepath`.

        Returns:
            Path to where the file is saved.
        """
        raise Exception("Not implemented!")


