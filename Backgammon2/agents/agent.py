"""
Brain naming convention
{agent config. type}-{agent config. name}-{timestamp}[-...]
"""

from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.nn_agent_1 import NNAgent1

from pathlib import Path
from lib.utils import hash_string, does_file_exist, hash_json, load_file_as_string, load_file_as_json, print_json
import os

agent_cfgs = {}

def load_agent_cfgs(dirname = "configs"):
    """
    Load all agent configs
    DONT IMPORT
    """
    agent_cfg_filenames = list(filter(lambda name: name[0:5] == 'agent' and name[-4:] == 'json', os.listdir(dirname)))
    for agent_cfg_filename in agent_cfg_filenames:
        filepath = str(Path(dirname, agent_cfg_filename))
        agent_cfg = load_file_as_json(filepath)
        name = agent_cfg['name']
        if name not in agent_cfgs:
            agent_cfgs[name] = agent_cfg
        else:
            raise Exception("At least two agent configs. share the same name: " + str(name))


# TODO: implement reload

# Yolo
def get_agent_config_by_config_name(config_name):
    return agent_cfgs[config_name]

def get_agent_by_config_name(config_name, brain_name = "new"):
    """
    Loads in agent by agent configuartion name.
    """
    # Agent configartion JSON.
    agent_cfg = get_agent_config_by_config_name(config_name)
    agent_cfg_name = agent_cfg['name']
    agent_cfg_type = agent_cfg['type']

    # TODO: implement load brain
    # Brain name
    agent = None
    if agent_cfg_type == "random":
        agent = RandomAgent()
    elif agent_cfg_type == "human":
        agent = HumanAgent()
    elif agent_cfg_type == "nn1":
        if brain_name == "new":
            agent = NNAgent1(agent_cfg=agent_cfg)
        elif brain_name == "best":
            agent = NNAgent1(agent_cfg=agent_cfg, imported=True)
        else:
            raise Exception("Something is not right")
    elif agent_cfg_type == 'best_nn1':
        agent = NNAgent1(agent_cfg = agent_cfg, load_best=True)
    else:
        raise Exception('Unknown type of agent: ' + str(agent_cfg_type))

    return agent

load_agent_cfgs()
