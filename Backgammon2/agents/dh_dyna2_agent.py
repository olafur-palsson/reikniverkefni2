#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon

class DHDyna2Agent(AgentInterface):

    def __init__(self, training = False):
        AgentInterface.__init__(self, training)
        self.training = training
        self.training_type = AgentInterface.TRAINING_TYPE_ONLINE
    
