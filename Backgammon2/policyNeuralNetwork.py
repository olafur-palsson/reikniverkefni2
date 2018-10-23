

from policy import Policy
from neuralNetwork import BasicNetworkForTesting
from parallelNetwork import ParallelNetwork
import random




def e_greedy(n):
    return random.randint(0, n)

class PolicyNeuralNetwork(Policy):


    number_of_decisions_0 = 0
    counter = 0
    epsilon = 0.15
    net = 0

    def __init__(self):
        self.net = ParallelNetwork()


    def evaluate(self, possible_boards):

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
                i = i + 1

        last_index_of_boards = len(possible_boards) - 1
        best_move = max_i
        move = best_move
        self.number_of_decisions_0 += int(move == 0)
        self.counter += 1
        # move = best_move if random.random() > self.epsilon else e_greedy(last_index_of_boards)
        self.net.run_decision(self.get_feature_vector(possible_boards[move]))
        return move

    def log_and_reset_no_zeros(self):
        print("")
        print("% of decisions '0' (first of array), lower is better ")
        print(str(float(self.number_of_decisions_0) / self.counter))
        self.number_of_decisions_0 = 0
        self.counter = 0

    def get_reward(self, reward):
        self.net.get_reward(reward)
        self.log_and_reset_no_zeros()
