
import numpy as np
import torch
import random

from policy import Policy

# example of an extended policy
class PolicyRandom(Policy):
    def get_file_name(self):
        return "lol"

    def evaluate(self, possible_boards):
        for board in possible_boards:
            feature_vector = self.get_feature_vector(board)
        return random.randint(len(possible_boards) - 1)

    def get_reward(self, reward):
        print("lol")
