

from policy import Policy
from neuralNetwork import NeuralNetwork

class PolicyNeuralNetwork(Policy):

    net = 0

    def __init__():
        self.net = new NeuralNetwork()


    # unfinished, move to NeuralNetworkAgent
    def evaluate(possible_boards):
        move_ratings = map(lambda board : net.evaluate(get_feature_vector(board)), possible_boards)
        best_move = move_ratings.index(max(move_ratings))
        return best_move

    # ! BROKEN, todo: FIX
    def get_reward(reward):
        net.zero_grad()
        net.get_reward(reward)


