
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.nn_agent_1 import NNAgent1
from agents.nn_agent_best_so_far import BestNNAgent


def get_agent(cfg):

    print("vvvvvvvvvvvvvv")

    agent = None

    if cfg['type'] == "random":
        agent = RandomAgent()
    elif cfg['type'] == "human":
        agent = HumanAgent()
    elif cfg['type'] == "nn1":
        agent = NNAgent1()
    elif cfg['type'] == 'best_nn1':
        agent = BestNNAgent()
    else:
        raise Exception('Unknown type of agent: ' + str(cfg['type']))

    print(cfg)

    print("^^^^^^^^^^^^^^")

    return agent
