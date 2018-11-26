#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import torch

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon


from torch.autograd import Variable

device = None

try:
    device = torch.device('cuda') 
except:
    device = torch.device('cpu')


class DHPGAgent(AgentInterface):

    def __init__(self, training = False):
        AgentInterface.__init__(self, training)
        self.training = training
        self.training_type = AgentInterface.TRAINING_TYPE_ONLINE

        # Set up neural network

    def action(self, board, dice, player):
        """
        Args:
            board (ndarray): backgammon board
            dice (ndarray): a pair of dice
            player: the number for the player on the board who's turn it is.

        Returns:
            A move `move`.
        """

        move, xtheta = self.softmax_policy(board, dice, player)

        if True:
            return move

        possible_moves, possible_boards = Backgammon.get_all_legal_moves_for_two_dice(board, dice)

        move = self.pick_move(possible_moves, possible_boards, dice)

        return move


    def pick_move(self, possible_moves, possible_boards, dice):

        move = []

        if len(possible_moves) == 0:
            return move

        # Find index of best board/move.
        index_of_best_board = self.evaluate(possible_boards)

        # Pick that move.
        move = possible_moves[index_of_best_board]

        return move



    def softmax_policy(self, board, dice, player):
        """
        Args:
            board: current board that is being evaluated.
            player: who this player is.

        Returns:
            A move, and something else
        """

        # 928
        size_feature_vector = len(self.get_one_hot_feature_vector(board))

        nr_of_boards = len(boards)

        # Get all possibles moves and their corresponding boards.
        moves, boards = Backgammon.get_all_legal_moves_for_two_dice(board, dice)

        if len(moves) == 0:
            # PASS
            return None


        # Like a matrix.  Each column vector corresponds to the board
        # in how one encoding...
        one_hot_boards = np.zeros((size_feature_vector, nr_of_boards))

        print(one_hot_boards)

        for i, board in enumerate(boards):
            # Treat this as column vectors
            one_hot_board = self.get_one_hot_feature_vector(board)

            print("One hot board")
            print(one_hot_board)

            one_hot_boards[:,i] = one_hot_board
            
        print("One hot boards")
        print(one_hot_boards)

        x = Variable(torch.tensor(one_hot_boards, dtype = torch.float, device = device))

        # RUN NEURAL NETWORK OHM NOM NOM.


        

        # Feature vector board

        idx = -1

        xtheta_mean = None

        if True:
            raise Exception("pause!")


        return moves[idx], xtheta_mean

    def evaluate(self, possible_boards):
        # variable to hold ratings
        move_ratings = []

        # predict win_rate of each possible after-state (possible_boards)
        for board in possible_boards:
            value_of_board = self.net.predict(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        # get max value
        max = move_ratings[0]
        max_i = 0
        for i, move in enumerate(move_ratings):
            if move > max:
                max = move
                max_i = i


        best_move = max_i
        move = best_move
        self.number_of_decisions_0 += int(move == 0)
        self.decision_counter += 1
        # move = best_move if random.random() > self.epsilon else random.rand_int(len(possible_boards - 1)) # uncomment for e_greedy
        self.net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move

    def add_reward(self, reward):
        """
        Adds reward `reward`, e.g. to current episode.

        Args:
            reward (number): the reward
        """
        raise Exception("Not implemented!")

    
    def train(self):
        pass

    def get_one_hot_feature_vector(self, board):
        """
        Returns a 1D float array.

        Args:
            board: array
        """

        features = np.array([])

        # This player
        for i in range(len(board)):
            hosv = np.zeros(16)
            n = int(board[i])
            if n > 0:
                hosv[n] = 1
            elif n < 0:
                pass
            elif n == 0:
                pass
            else:
                raise Exception("Shouldn't have happened!")
            features = np.append(features, hosv)
        
        # Opponent
        for i in range(len(board)):
            hosv = np.zeros(16)
            n = int(board[i])
            if n > 0:
                pass
            elif n < 0:
                hosv[-n] = 1
            elif n == 0:
                pass
            else:
                raise Exception("Shouldn't have happened!")
            features = np.append(features, hosv)
        
        # Cast every entry to float.
        # features = torch.from_numpy(features).float()

        # Requires grad.
        # features.requires_grad = True

        return features
