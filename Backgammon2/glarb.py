from trueskill import Rating, quality_1vs1, rate_1vs1
from pathlib import Path
from lib.utils import load_file_as_json, does_file_exist, save_json_to_file, hash_json, timestamp
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


def rating_to_string(rating):
    """
    Helper function
    """

    fmt_s = '{0:.6f}'

    mu = rating.mu
    sigma = rating.sigma

    return "µ = " + str.format(fmt_s, mu) + ", σ = " + str.format(fmt_s, sigma)


def random_pair_not_self(arr):
    """
    Garbage function
    """
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




def print_competitors(competitors):
    print("Rating of each player")
    print("")
    competitors.sort(key=lambda competitor: competitor['rating'].mu - competitor['rating'].sigma)

    for i, competitor in enumerate(competitors):
        rating = competitor['rating']
        name = competitor['cfg']['name']

        print("Player " + str(i + 1) + " ("  + name + "): ")
        print("    TrueSkill: " + rating_to_string(rating))
        print("    Played games: " + str(competitor['played_games']))
        print("    Wins/losses: " + str(competitor['wins']) + " / " + str(competitor['losses']))
        _wlr = float('inf') if competitor['losses'] == 0 else competitor['wins'] / competitor['losses']
        print("    Win/Loss Ratio: " + str(_wlr))



def save_competitors(competition, competitors):
    print("Saving...")

    manifest = Manifest("./repository/manifest.json")
    manifest.load()
    

    competition_config_hash = hash_json(competition)

    competition_result = {
        "competitor_result_hashes": []
    }
    
    for competitor in competitors:
        
        agent_config_name = competitor['agent_config_name']
        agent_config = competitor['cfg']
        agent = competitor['agent']


        agent_config_hash = hash_json(competitor["cfg"])

        print(competitor['agent'])

        brain_location = agent.save()
        print("> " + str(brain_location))

        competitor_result = {
            "trueskill_rating": {
                "mu": competitor['rating'].mu,
                "sigma": competitor['rating'].sigma
            },
            "wins": competitor['wins'],
            "losses": competitor['losses'],
            "competition_config_hash": competition_config_hash,
            "timestamp": timestamp(),
            "agent_config_hash": agent_config_hash,
            "played_games": competitor["played_games"]
        }

        

        if brain_location:
            competitor_result["brain_location"] = brain_location

        competitor_result_hash = hash_json(competitor_result)

        competition_result["competitor_result_hashes"] += [competitor_result_hash]

        manifest.set("competitor_result{}" + competitor_result_hash, competitor_result)
        manifest.set("agent{}" + agent_config_hash + "{}competitions[]+", competitor_result_hash)

        manifest.set("agent_config{}" + agent_config_hash + "{}name", agent_config_name)


        try:
            best_trueskill = manifest.get("agent_config{}" + agent_config_hash + "{}best{}trueskill")

            a = competitor_result["trueskill_rating"]["mu"] - competitor_result["trueskill_rating"]["sigma"]
            b = best_trueskill["mu"] - best_trueskill["sigma"]

            if a > b:
                manifest.set("agent_config{}" + agent_config_hash + "{}best{}trueskill", competitor_result["trueskill_rating"])
                manifest.set("agent_config{}" + agent_config_hash + "{}best{}competitor_result_hash", competitor_result_hash)
        except:
            manifest.set("agent_config{}" + agent_config_hash + "{}best{}trueskill", competitor_result["trueskill_rating"])
            manifest.set("agent_config{}" + agent_config_hash + "{}best{}competitor_result_hash", competitor_result_hash)


    competition_result_hash = hash_json(competition_result)

    competition_hash = hash_json(competition)

    manifest.set("competition{}" + competition_hash + "{}results{}" + competition_result_hash, competition_result)
    
    manifest.save()


def do_glarb(path_to_competition_json = "configs/competition_test.json"):

    print("Loading competition: " + str(path_to_competition_json))
    
    # Load in competition.
    competition = load_file_as_json(path_to_competition_json)

    # Get competitors information.
    competitors_info = competition["competitors"]

    
    # Load in competitors.
    competitors = []
    for competitor_info in competitors_info:

        # Get agent configuration.
        agent_config_name = competitor_info["cfg"]

        # Determine which "brain" to load in, e.g. which neural network.
        brain = competitor_info["brain"] if "brain" in competitor_info else "new"

        # Acquire the agent configuration JSON.
        agent_config = get_agent_config_by_config_name(agent_config_name)

        # Load in the agent.
        agent = get_agent_by_config_name(agent_config_name, brain)

        # Create a container for the competitor.
        competitor = {
            "cfg": agent_config,
            "agent": agent,
            "rating": Rating(25, 25/3),
            "played_games": 0,
            "losses": 0,
            "wins": 0,
            "agent_config_name": agent_config_name,
            "brain": brain
        }

        competitors += [competitor]

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
                print_competitors(competitors)

            if iteration % 10000 == 0:
                save_competitors(competition, competitors)
    except:
        print("Halted!")
    
    print_competitors(competitors)
    save_competitors(competition, competitors)



    print("Exiting...")
