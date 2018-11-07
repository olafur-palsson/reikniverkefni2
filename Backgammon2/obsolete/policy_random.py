#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import torch
import random

from policy import Policy

# example of an extended policy
class PolicyRandom(Policy):
    def get_filename(self):
        return "lol"

    def evaluate(self, possible_boards):
        return random.randint(0, len(possible_boards) - 1)

    def get_reward(self, reward):
        print("lol")
