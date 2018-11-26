
import numpy as np
import random

from pub_stomper_policy import Policy
from pub_stomper_basic_network_for_testing import BasicNetworkForTesting
from parallel_network import ParallelNetwork

class PolicyPsuedo(Policy):

    def __init__(self, neural_net, should_update=False, verbose=False, agent_cfg=None, imported=False, pub_stomper_policy_decision_function='argmax'):
        """
        Args:
            load_best (bool): default `False`
            verbose (bool): default `False`
            export (bool): default `False`
            agent_cfg: default `None`
            archive_name: default `None`.
        """

        self.verbose = verbose
        self.net = neural_net

        self.should_update = should_update

    def argmax(self, move_ratings):
        # get max value
        max = move_ratings[0]
        max_i = 0
        for i, move in enumerate(move_ratings):
            if move > max:
                max = move
                max_i = i
        return max_i

    def evaluate(self, possible_boards):
        """
        Evaluates the possible boards given to this method as an argument and
        returns a move.

        Args:
            possible_boards: possible boards

        Returns:
            A move.
        """
        # variable to hold ratings
        move_ratings = []

        # predict win_rate of each possible after-state (possible_boards)
        for board in possible_boards:
            value_of_board = self.net.predict(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        move = 0
        # move = best_move if random.random() > self.epsilon else random.rand_int(len(possible_boards - 1)) # uncomment for e_greedy
        self.net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move

    def save(self, save_as_best=False):
        return self.net.save(save_as_best=save_as_best)

    def load(self, filename):
        self.net.load(filename)

    def get_filename(self):
        """
        Returns the file name for this neural network attached to this instance.

        Returns:
            The file name of the neural network.
        """
        return self.net.filename

    def add_reward(self, reward):
        if self.should_update:
            self.net.give_reward_to_nn(reward)
