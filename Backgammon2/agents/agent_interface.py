#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_interface.py

A class, or interface, for agents. 
"""

class AgentInterface():
    """
    A standardized interface for agents.
    """

    # Train after each episode.
    TRAINING_TYPE_ONLINE = 1
    
    # Train after N episodes.  Accumulate results.
    TRAINING_TYPE_MINI_BATCH = 2

    # Train after completing all episodes.
    TRAINING_TYPE_BATCH = 3

    def __init__(self, training = False):
        """
        Instantiates an standardized agent interface.

        Args:
            training (bool): whether this agent is training, default `False`
        """
        self.training = training
        self.training_type = AgentInterface.TRAINING_TYPE_ONLINE


    def set_training_type(self, training_type):
        """
        Denotes how our training schedule should be like.

        * Online: train after each episode.
        * Mini-batch: train after some fixed number of episodes.
        * Batch: train once we have accumulated all episodes.

        Args:
            training_type (number): training schedule
        """
        assert training_type == AgentInterface.TRAINING_TYPE_ONLINE or training_type == AgentInterface.TRAINING_TYPE_MINI_BATCH or training_type == AgentInterface.TRAINING_TYPE_BATCH

        self.training_type = training_type


    def action(self, board, dice, player):
        """
        This method takes in a board, dice and the player's number and
        returns a "move".

        A "move" is a list, where each entry is a list of two numbers, and
        corresponds to a roll of the dice.  The sublist of the "move" denotes 
        where we pick up the checker and and where we put it.

        Example output:

            [ [18, 16], [16, 10] ]

        We move a checker from 18 to 16 and a checker from 16 to 10.

        NOTE: I think the order matters.

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
        Adds action `action`, e.g. to current episode.

        Args:
            action: the action.
        """
        raise Exception("Not implemented!")


    def add_reward(self, reward):
        """
        Adds reward `reward`, e.g. to current episode.

        Args:
            reward (number): the reward
        """
        raise Exception("Not implemented!")
    

    def add_state(self, state):
        """
        Adds state `state`, e.g. to current episode.

        Args:
            state: the state
        """
        raise Exception("Not implemented!")


    def load(self, filepath = None):
        """
        Loads agent from disk at `filepath`.

        NOTE: Refrain from using `filepath`.

        Returns:
            Path to where the file is saved.
        """
        raise Exception("Not implemented!")
    

    def save(self, filepath = None):
        """
        Saves agent to disk at `filepath`.

        NOTE: Refrain from using `filepath`, as the object that implements
        this feature is in control of saving.  I don't know...

        Returns:
            Path to where the file is saved.
        """
        raise Exception("Not implemented!")

    def pre_game(self):
        pass

    def post_game(self):
        pass


