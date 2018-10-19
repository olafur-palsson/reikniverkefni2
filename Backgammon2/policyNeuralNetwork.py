

from policy import Policy
from neuralNetwork import NeuralNetwork

class PolicyNeuralNetwork(Policy):

    net = 0

    def __init__(self):
        self.net = NeuralNetwork()


    def evaluate(self, possible_boards):

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
        best_move = max_i

        self.net.run_decision(self.get_feature_vector(board))
        return best_move

    def get_reward(self, reward):
        self.net.get_reward(reward)
