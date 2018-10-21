

from policy import Policy
from neuralNetwork import NeuralNetwork, BasicNetworkForTesting
import random


epsilon = 0.05

def e_greedy(n):
    return random.randint(0, n)

class PolicyNeuralNetwork(Policy):

    net = 0

    def __init__(self):
        self.net = BasicNetworkForTesting()


    def evaluate(self, possible_boards):
        epsilon = 0.1
        move_ratings = []
        for board in possible_boards:
            value_of_board = self.net.predict(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        max = move_ratings[0]
        max_i = 0
        i = 0
        for move in move_ratings:
            if move > max:
                max = move
                max_i = i

        last_index_of_boards = len(possible_boards) - 1
        move = max_i
        # move = best_move if random.random() > epsilon else e_greedy(last_index_of_boards)
        self.net.run_decision(self.get_feature_vector(board))
        return best_move

    def get_reward(self, reward):
        self.net.get_reward(reward)
