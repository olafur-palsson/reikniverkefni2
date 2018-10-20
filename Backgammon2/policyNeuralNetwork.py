

from policy import Policy
from neuralNetwork import NeuralNetwork
import random



class PolicyNeuralNetwork(Policy):

    net = 0

    def __init__(self):
        self.net = NeuralNetwork()


    def evaluate(self, possible_boards):
        epsilon = 0.1
        move_ratings = []
        for board in possible_boards:
            value_of_board = self.net.evaluate(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        max = move_ratings[0]
        max_i = 0
        i = 0
        for move in move_ratings:
            if move > max:
                max = move
                max_i = i

        last_index_of_boards = len(possible_boards) - 1

        best_move = max_i
        # best_move = random.randint(0, last_index_of_boards) if random.uniform(-1, 1) > 0 else best_move
        self.net.run_decision(self.get_feature_vector(board))
        return best_move

    def get_reward(self, reward):
        self.net.get_reward(reward)
