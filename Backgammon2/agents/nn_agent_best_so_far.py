from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon
from policy_neural_network import PolicyNeuralNetwork
from agents.nn_agent_1 import NNAgent1



class BestNNAgent(NNAgent1):
        
    def __init__(self, verbose = False, agent_cfg = None):
        """
        Args:
            verbose: default `False`
            agent_cfg: default `None`
        """
        NNAgent1.__init__(self, load_best=True, verbose = verbose, agent_cfg = agent_cfg)
        # super().__init__(self, load_best=True, verbose = verbose, agent_cfg = agent_cfg)
        
