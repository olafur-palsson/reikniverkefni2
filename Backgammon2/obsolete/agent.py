#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the agent
"""
import numpy as np
import Backgammon
from policy_neural_network import PolicyNeuralNetwork
# from policy_random import PolicyRandom

pub_stomper = PolicyNeuralNetwork()

# muna ad setja optional parameter fyrir flipped board gaejann

def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    move = []

    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice)

    # make the best move according to the policy
    if len(possible_moves) != 0:
        move = policy(possible_moves,possible_boards,dice,i)

    return move


def reward_player(reward):
    pub_stomper.get_reward(reward)


def get_file_name():
    return pub_stomper.get_file_name()


def policy(possible_moves, possible_boards, dice, i):

    best_move = pub_stomper.evaluate(possible_boards)
    move = possible_moves[best_move]

    # gamli kodinn fyrir random
    # move = possible_moves[np.random.randint(len(possible_moves))]
    return move
