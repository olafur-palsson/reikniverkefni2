

from policy import Policy
from neuralNetwork import NeuralNetwork

class PolicyNeuralNetwork(Policy):

    net = 0

    def __init__(self):
        self.net = NeuralNetwork()


    def evaluate(self, possible_boards):

        move_ratings = []
        move_ratings2 = map(lambda board : self.net.evaluate(self.get_feature_vector(board)), possible_boards)
        should_print = True
        for board in possible_boards:
            value_of_board = self.net.evaluate(self.get_feature_vector(board))
            print(value_of_board)
            move_ratings.append(value_of_board)
            should_print = False
        print("")
        print("policy -> evaluate")
        print("move ratings")

        best_board, best_move = move_ratings.max(0)
        self.net.run_decision(board)
        return best_move

    def get_reward(self, reward):
        net.zero_grad()
        net.get_reward(reward)
