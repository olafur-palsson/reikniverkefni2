
import numpy as np
import torch
import random

from policy import Policy

def e_greedy(n):
    return random.randint(0, n)

class PolicyRandom(Policy):

    

    def get_file_name(self):
        return "lol"

    def evaluate(self, possible_boards):

        for board in possible_boards:
            feature_vector = self.get_feature_vector(board)




        return e_greedy(len(possible_boards) - 1)

    def get_reward(self, reward):
        print("lol")
