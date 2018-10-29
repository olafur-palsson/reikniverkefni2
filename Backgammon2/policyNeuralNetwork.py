

from policy import Policy
from neuralNetwork import BasicNetworkForTesting
from parallelNetwork import ParallelNetwork
import numpy as np
import random




class PolicyNeuralNetwork(Policy):

    # Epsilon for e-greedy
    epsilon = 0.15

    # Data for statistics
    number_of_decisions_0 = 0
    decision_counter = 0
    counter = 0
    net = 0
    last_500 = np.zeros(500)

    # Decide what neural network to use
    # self.net = BasicNetworkForTesting()
    # or
    # self.net = ParallelNetwork() <-- little crazy
    def __init__(self):
        self.net = BasicNetworkForTesting()

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

    def get_file_name(self):
        return self.net.file_name

    def log_and_reset_number_of_decisions_0(self):
        print("")
        print("% of decisions '0' (first of array), lower is better ")
        print(str(float(self.number_of_decisions_0) / self.decision_counter))
        self.number_of_decisions_0 = 0
        self.decision_counter = 0

    def get_reward(self, reward):
        # only necessary line in this function
        self.net.get_reward(reward)

        # statistics
        self.last_500[self.counter % 500] = reward
        self.counter += 1
        exp_return = np.sum(self.last_500) / 500 # this is from -1 to 1
        print("")
        print("Expected return")
        print(exp_return)
        self.log_and_reset_number_of_decisions_0()
