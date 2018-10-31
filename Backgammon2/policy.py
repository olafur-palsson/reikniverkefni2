#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I think this class, `Policy`, should be extended.
"""

import numpy as np
import torch

# Extend this class to make a policy to have all the feature_vector functions
class Policy():

    def get_tesauro_feature_vector(self, board):
        """
        This returns a feature vecture as is used by Tesauro in
        TD-Gammon.

        Args:
            board (ndarray): a backgammon board

        Returns:
            A feature vector.
        """
        main_board = board[1:25]
        jail1, jail2, off1, off2 = board[25], board[26], board[27], board[28]
        features = np.array([])

        # naum i feature vector af adalsvaedinu
        for position in main_board:
            vector = np.zeros(4)
            sign = -1 if position < 0 else 1
            for i in range(int(abs(position))):
                if i > 3:
                    vector[3] = sign * (abs(position) - 3) / 2
                    break
                vector[i] = position/abs(position)
            features = np.append(features, vector)

        # jail feature-ar
        jail_features = np.array([jail1, jail2]) * 0.5

        # features fyrir hversu margir eru borne off
        off_board_features = np.array([off1, off2]) * (0.066667)
        bias_vector = np.array([1, 1])
        features =  np.append(features, [jail_features, off_board_features, bias_vector])
        features = torch.from_numpy(features).float()
        features.requires_grad = True
        return features


    def get_feature_vector(self, board):
        """
        Returns the raw feature vector for the backgammon board `board`.

        Args:
            board (ndarray): a backgammon board

        Returns:
            A feature vector.
        """
        return self.get_raw_data(board)
        # return self.get_tesauro_feature_vector(self, board)


    # expand board -> 464 vector
    def get_raw_data(self, board):
        """
        Returns the raw feature vector for the backgammon board `board`.

        Args:
            board (ndarray): a backgammon board

        Returns:
            A feature vector.
        """
        features = np.array([])
        for position in board:
            vector = np.zeros(16)
            for i in range(int(position)):
                vector[i] = 1
            features = np.append(features, vector)
        features = torch.from_numpy(features).float()
        features.requires_grad = True
        return features

    # Override these methods
    def add_reward(self, reward):
        raise Exception("Reward function not set")

    def evaluate(self, board):
        raise Exception("Evaluation function not set")

    def get_file_name(self):
        raise Exception("File name not set")
