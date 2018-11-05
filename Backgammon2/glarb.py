
# Experimental
# $ pip install trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1


from pathlib import Path

from agents.agent import get_agent

from lib.utils import load_file_as_json
from backgammon_game import Backgammon

import os


def update_rating(rating1, rating2, result):
    """
    Updates rating for player 1 (`rating1`) and rating for player 2 (`rating2`).

    Args:
        rating1: rating for player 1
        rating2: rating for player 2
        result: +1 if player 1 won, 0 if there was a tie, and -1 is player 2 won.

    Returns:
        Updated (rating1', rating2')
    """

    new_rating1 = None
    new_rating2 = None

    if result == 1:
        # Player 1 won
        new_rating1, new_rating2 = rate_1vs1(rating1, rating2)
    elif result == 0:
        # A tie
        new_rating1, new_rating2 = rate_1vs1(rating1, rating2, drawn=True)
    elif result == -1:
        # Player 2 won
        new_rating2, new_rating1 = rate_1vs1(rating2, rating1)
    else:
        raise Exception("This shouldn't have happened!", result)

    return (new_rating1, new_rating2)


def test2():
    # Assign Alice and Bob's ratings
    alice = Rating(mu=25.000, sigma=8.333)
    bob = Rating(mu=25.000, sigma=8.333)


    alice2, bob2 = update_rating(alice, bob, 0)
    
    print(alice, bob)
    print(alice2, bob2)



def rating_to_string(rating):

    fmt_s = '{0:.6f}'

    mu = rating.mu
    sigma = rating.sigma

    return "µ = " + str.format(fmt_s, mu) + ", σ = " + str.format(fmt_s, sigma)


def ts_test():
    player1 = RandomAgent()
    player2 = BestNNAgent()

    rating1 = Rating()
    rating2 = Rating()

    # Play n games

    n = 100

    print("P1: " + str("random"))
    print("P2: " + str("best NN"))

    i = 0
    print(str(i) + ": P1 (" + rating_to_string(rating1) + "), P2(" + rating_to_string(rating2) + ")")

    while True:
        print('{:.1%} chance to draw'.format(quality_1vs1(rating1, rating2)))
        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        result = bg.play()
        print(result)
        rating1, rating2 = update_rating(rating1, rating2, result)
        i += 1
        print(str(i) + ": P1 (" + rating_to_string(rating1) + "), P2(" + rating_to_string(rating2) + ")")


agent_configs = {}


def load_agent_configs():
    dirname = "configs"
    agent_config_filenames = list(filter(lambda name: name[0:5] == 'agent' and name[-4:] == 'json', os.listdir(dirname)))

    for agent_config_filename in agent_config_filenames:
        filepath = str(Path(dirname, agent_config_filename))
        agent_config = load_file_as_json(filepath)
        name = agent_config['name']
        if name not in agent_configs:
            agent_configs[name] = agent_config
        else:
            raise Exception("At least two agent configs. share the same name: " + str(name))


def do_glarb():

    # Load in agent configurations
    load_agent_configs()

    

    competitors_info = load_file_as_json("configs/playerbase_test.json")["competitors"]
    # Load in competitors
    
    competitors = []
    for competitor_info in competitors_info:
        agent_config_name = competitor_info["cfg"]

        agent_config = agent_configs[agent_config_name]
        agent = get_agent(agent_config)

        competitior = {
            "cfg": agent_config,
            "agent": agent
        }

        competitors += [competitior]

    print(competitors)




    pass