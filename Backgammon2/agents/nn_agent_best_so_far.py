



# Hugmyndin er ad hafa thennan sem auto-load-a besta playernum
#

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon
from policy_neural_network import PolicyNeuralNetwork
from nn_agent_1.py import NNAgent1

class BestNNAgent(NNAgent1):
    def __init__(self, verbose=False):
        super().__init__()
        self.policy = PolicyNeuralNetwork(load_best=True, verbose=verbose)
