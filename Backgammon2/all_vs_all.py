

# do_glarb with no manifest.json

import os
import numpy as np
from pathlib import Path
from lib.utils import load_file_as_json
from backgammon_game import Backgammon
from agents.agent import get_agent_config_by_config_name, get_agent_by_config_name

from trueskill import Rating, quality_1vs1, rate_1vs1

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

    # (result == 1) means player1 won else player2 won
    if   result == 1:  new_rating1, new_rating2 = rate_1vs1(rating1, rating2)
    elif result == -1: new_rating2, new_rating1 = rate_1vs1(rating2, rating1)
    else:
        raise Exception("This shouldn't have happened!", result)

    return (new_rating1, new_rating2)

def rating_to_string(rating):
    fmt_s = '{0:.6f}'
    mu = rating.mu
    sigma = rating.sigma
    return "µ = " + str.format(fmt_s, mu) + ", σ = " + str.format(fmt_s, sigma)

def random_pair_not_self(array):
    """
    Returns:
        2 distinct elements from array
    """
    n = len(array)
    if n <= 1:
        raise Exception("Not enough players, if you're using 2, use ")
    while True:
        index_player1, index_player2 = np.random.randint(0, n), np.random.randint(0, n)
        if not index_player1 == index_player2:
            return array[index_player1], array[index_player2]


def update_wins_and_losses(result, competitor1, competitor2):
    competitor1['played_games'] += 1
    competitor2['played_games'] += 1
    if result > 0:
        competitor1['wins'] += 1
        competitor2['losses'] += 1
    else:
        competitor1['losses'] += 1
        competitor2['wins'] += 1

def print_competitors(competitors, iteration):
    print("")
    print("")
    print("State at game number: " + str(iteration))
    print("")
    print("Rating of each player")
    print("")
    # sort players highest skill to lowest
    competitors.sort(key=lambda competitor: competitor['rating'], reverse=True)

    for i, competitor in enumerate(competitors):
        rating = competitor['rating']
        name = competitor['cfg']['name']
        print("Player " + str(i + 1) + " ("  + name + "): ")
        print("    TrueSkill: " + rating_to_string(rating))
        print("    Played games: " + str(competitor['played_games']))
        print("    Wins/losses: " + str(competitor['wins']) + " / " + str(competitor['losses']))
        _wlr = float('inf') if competitor['losses'] == 0 else competitor['wins'] / competitor['losses']
        print("    Win/Loss Ratio: " + str(_wlr))



def save_competitors(competitors):
    # TODO: Gera thetta thannig ad thad save-ar alltaf besta version
    print("Saving...")
    competitor_names_already_saved = ['nn_best']
    best_network_is_playing = False
    # Save all to their corresponding place in './repository'
    for competitor in competitors:
        name_of_competitor = competitor['cfg']['name']
        # BETA: Veit ekki hvort thetta virkar enntha
        # Kannski overwrite-ar thetta besta ef besta er ekki a stadnum
        if name_of_competitor == 'nn_best':
            best_network_is_playing = True
        # Check if we already saved a competitor with higher rating
        already_saved = False
        for saved_competitor_name in competitor_names_already_saved:
            if name_of_competitor == saved_competitor_name:
                already_saved = True
        # We haven't save this one then
        if already_saved:
            continue
        competitor_names_already_saved.append(name_of_competitor)
        competitor['agent'].save()

    # if the best competitor is of type nn1 we export it as best
    if competitors[0]['cfg']['type'] == 'nn1' and best_network_is_playing:
        print('Best network was: ', competitors[0]['cfg']['name'])
        competitors[0]['agent'].save(save_as_best=True)

def train(competitors):
    # Train
    print("Training...")
    iteration = 0
    while True:
        iteration += 1
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
        update_wins_and_losses(result, competitor1, competitor2)

        # Rate performance
        competitor1['rating'], competitor2['rating'] = update_rating(competitor1['rating'], competitor2['rating'], result)

        if iteration % 10 == 0:
            print_competitors(competitors, iteration)

        if iteration % (100 * len(competitors)) == 0:
            save_competitors(competitors)


def make_competitor(competitor_info):
    agent_config_name = competitor_info['cfg']
    print(competitor_info)
    brain_type = competitor_info['brain']
    agent_cfg = get_agent_config_by_config_name(agent_config_name)
    agent = get_agent_by_config_name(agent_config_name, brain_type)
    return {
        "cfg": agent_cfg,
        "agent": agent,
        "rating": Rating(25, 25/3),
        "played_games": 0,
        "losses": 0,
        "wins": 0,
        "agent_config_name": agent_config_name,
    }


def play_all_vs_all(competition_cfg_file):
    if not competition_cfg_file: competition_cfg_file = 'competition_test'
    path = "configs/" + competition_cfg_file + '.json'

    #
    competition_setup = load_file_as_json(path)

    # Get competitors information.
    competitors = competition_setup["competitors"]

    def make_params(competitor_info):
        return [agent_config_name, agent_cfg, agent]

    competitors = list(map(lambda comp_info: make_competitor(comp_info), competitors))
    train(competitors)
    print_competitors(competitors, iteration)
    print("Exiting...")
