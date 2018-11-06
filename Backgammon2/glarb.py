
# Experimental
# $ pip install trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1


from pathlib import Path

from lib.utils import load_file_as_json, does_file_exist, save_json_to_file
from backgammon_game import Backgammon

import numpy as np

import os

from agents.agent import get_agent_config_by_config_name, get_agent_by_config_name


from lib.manifest import Manifest


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









def random_pair_indices_not_self(n):
    if n == 0:
        raise Exception("Something went wrong:" + str(n))
    elif n == 1:
        return (0, 0)
    if n > 1:
        idx1 = np.random.randint(0, n)
        idx2 = idx1
        while idx2 == idx1:
            idx2 = np.random.randint(0, n)
        return (idx1, idx2)
    raise Exception("Something went wrong:" + str(n))

def random_pair_not_self(arr):
    if len(arr) == 0:
        raise Exception("Shouldn't happen!")
    elif len(arr) == 1:
        return (arr[0], arr[0])
    if len(arr) > 1:
        n = len(arr)
        idx1 = np.random.randint(0, n)
        idx2 = idx1
        while idx2 == idx1:
            idx2 = np.random.randint(0, n)
        return (arr[idx1], arr[idx2])
    raise Exception("Shouldn't happen!")




        






def do_glarb():

    competition = load_file_as_json("configs/competition_test.json")
    competitors_info = competition["competitors"]

    # Load in information about competitors.
    
    # Load in competitors
    competitors = []
    for competitor_info in competitors_info:

        agent_config_name = competitor_info["cfg"]
        brain = competitor_info["brain"] if "brain" in competitor_info else "new"

        agent_config = get_agent_config_by_config_name(agent_config_name)
        agent = get_agent_by_config_name(agent_config_name, brain)

        

        competitior = {
            "cfg": agent_config,
            "agent": agent,
            "rating": Rating(25, 25/3),
            "played_games": 0,
            "losses": 0,
            "wins": 0,
            "agent_config_name": agent_config_name
        }

        competitors += [competitior]

    # Train
    
    print("Training...")
    
    iteration = 0
    try:
        while True:
            iteration += 1
            print(str(iteration))

            competitor1, competitor2 = random_pair_not_self(competitors)

            player1 = competitor1['agent']
            player2 = competitor2['agent']

            player1.training = True
            player2.training = True

            bg = Backgammon()
            bg.set_player_1(player1)
            bg.set_player_2(player2)

            # 1 if player 1 won, -1 if player 2 won
            result = bg.play()

            player1.add_reward(result)
            player2.add_reward(-result)

            competitor1['played_games'] += 1
            competitor2['played_games'] += 1

            if result == 1:
                # Player 1 won
                # Player 2 lost
                competitor1['wins'] += 1
                competitor2['losses'] += 1
            elif result == -1:
                # Player 1 lost
                # Player 2 won
                competitor1['losses'] += 1
                competitor2['wins'] += 1
            else:
                raise Exception("Unexpected result: " + str(result))

            # Rate performance
            rating1, rating2 = competitor1['rating'], competitor2['rating']
            competitor1['rating'], competitor2['rating'] = update_rating(rating1, rating2, result)

            if iteration % 10 == 0:
                print("Rating of each player")
                print("")

                # Sort competitors by their TrueSkill rating.
                competitors.sort(key=lambda competitor: competitor['rating'])

                for i, competitor in enumerate(competitors):
                    rating = competitor['rating']
                    name = competitor['cfg']['name']

                    print("Player " + str(i + 1) + " ("  + name + "): ")
                    print("    TrueSkill: " + rating_to_string(rating))
                    print("    Played games: " + str(competitor['played_games']))
                    print("    Wins/losses: " + str(competitor['wins']) + " / " + str(competitor['losses']))
                    _wlr = float('inf') if competitor['losses'] == 0 else competitor['wins'] / competitor['losses']
                    print("    Win/Loss Ratio: " + str(_wlr))

    except:
        print("Halted!")

    print("Rating of each player")
    print("")

    # Sort competitors by their TrueSkill rating.
    competitors.sort(key=lambda competitor: competitor['rating'])

    for i, competitor in enumerate(competitors):
        rating = competitor['rating']
        name = competitor['cfg']['name']

        print("Player " + str(i + 1) + " ("  + name + "): ")
        print("    TrueSkill: " + rating_to_string(rating))
        print("    Played games: " + str(competitor['played_games']))

    # print("Comparing players")

    print("Saving...")

    manifest = Manifest("./repository/manifest.json")
    manifest.load()
    
    for i, competitor in enumerate(competitors):
        
        agent_config_name = competitor['agent_config_name']
        agent_config = competitor['cfg']
        agent = competitor['agent']

        print(competitor['agent'])

        filename = agent.save()
        print("> " + str(filename))

        if filename:

            info = {
                "rating": {
                    "mu": competitor['rating'].mu,
                    "sigma": competitor['rating'].sigma
                },
                "wins": competitor['wins'],
                "losses": competitor['losses'],
                "agent_config_name": competitor['agent_config_name']
            }

            manifest.set(
                "filenames{}" + filename + "{}log[]+",
                info
            )
    
    manifest.save()


    print("Exiting...")