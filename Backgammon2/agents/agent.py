"""

Brain naming convention


{agent config. type}-{agent config. name}-{timestamp}[-...]

"""


from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.nn_agent_1 import NNAgent1


from pathlib import Path
from lib.utils import load_file_as_json
from lib.utils import load_file_as_string
from lib.utils import hash_string, does_file_exist
import os



agent_configs_source = {}
agent_configs = {}




def get_brain_name(agent_config_type, agent_config_name, timestamp):
    return None





def load_agent_configs(dirname = "configs"):
    """
    Load all agent configs

    DONT IMPORT
    """

    dirname = "configs"
    agent_config_filenames = list(filter(lambda name: name[0:5] == 'agent' and name[-4:] == 'json', os.listdir(dirname)))
    
    for agent_config_filename in agent_config_filenames:
        filepath = str(Path(dirname, agent_config_filename))
        agent_config = load_file_as_json(filepath)
        agent_config_source = load_file_as_string(filepath)

        name = agent_config['name']

        if name not in agent_configs:
            agent_configs[name] = agent_config
            agent_configs_source[name] = agent_config_source
        else:
            raise Exception("At least two agent configs. share the same name: " + str(name))

# TODO: implement reload

def get_agent_config_by_config_name(config_name):

    parts = config_name.split(':')

    agent_config = None

    if len(parts) == 1:
        config_name = parts[0]
        agent_config = agent_configs[config_name]
        pass
    elif len(parts) == 2:
        config_name = parts[0]
        config_hash = parts[1]

        actual_config_hash = hash_string(agent_configs_source[config_name])

        if config_hash == actual_config_hash:
            agent_config = agent_configs[config_name]
        else:
            raise Exception("Agent doesn't exist")
    else:
        raise Exception("This shouldn't have happened!")

    return agent_config


def get_agent_by_config_name(config_name, brain_name = "new"):

    agent_config = get_agent_config_by_config_name(config_name)

    agent_config_name = agent_config['name']
    agent_config_type = agent_config['type']

    # TODO: implement load brain

    # Brain name

    agent = None

    if agent_config_type == "random":
        agent = RandomAgent()
    elif agent_config_type == "human":
        agent = HumanAgent()
    elif agent_config_type == "nn1":
        if brain_name == "new":
            agent = NNAgent1(agent_cfg = agent_config)
        elif brain_name == "best":
            agent = NNAgent1(agent_cfg = agent_config)
            print("Fetching best brain")
            if does_file_exist('./repository/manifest.json'):
                manifest = load_file_as_json('./repository/manifest.json')
                print(manifest)
            else:
                print(":'(")
                agent = NNAgent1(agent_cfg = agent_config)
        else:
            raise Exception("Something is not right")
    elif agent_config_type == 'best_nn1':
        agent = NNAgent1(agent_cfg = agent_config, load_best=True)
    else:
        raise Exception('Unknown type of agent: ' + str(agent_config['type']))

    return agent



load_agent_configs()
