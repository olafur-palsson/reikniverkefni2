#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point of this program.

To get default (old) behavior, run `python3 main.py default`.

Usage: python3 main.py ...
"""
import os
import sys
from pathlib import Path

import numpy as np

from cli_commands import get_commands_object

from backgammon_game import Backgammon
from pub_stomper_agents.human_agent import HumanAgent
from pub_stomper_agents.dyna_2_agent import Dyna2Agent
from pub_stomper_agents.random_agent import RandomAgent
from pub_stomper_agents.nn_agent_1 import NNAgent1

from pub_stomper_agents.agent import get_agent_by_config_name
from pub_stomper_lib.utils import hash_json, load_file_as_json
from statistic import Statistic
from pub_stomper_all_vs_all import play_all_vs_all

# Set logs
verbose = True

def do_default():
    """
    Play with a neural network against random
    """

    player1 = get_agent_by_config_name('nn_dyna2', 'new')
    player2 = RandomAgent()

    player1.training = True
    player2.training = True

    stats = Statistic(player1, verbose=True)

    # play games forever
    while True:

        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        winner = bg.play()

        player1.add_reward(winner)
        player2.add_reward(-winner)

        # Reward the neural network agent
        # player1.reward_player(winner)

        stats.add_win(winner)

        # Print out a log of game-stats
        if stats.games_played % 10 == 0:
            stats.output_result()


def nn_vs_nn_export_better_player():
    player1 = NNAgent1(verbose = True)
    player2 = NNAgent1(load_best=True)

    stats = Statistic(player1, verbose=True)

    while True:
        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        winner = bg.play()

        player1.add_reward(winner)
        player2.add_reward(-1 * winner)

        stats.add_win(winner)

        if stats.nn_is_better() and stats.games_played % 100 == 0:
            break

    # only way to reach this point is if the current
    # neural network is better than the BestNNAgent()
    # ... at least I think so
    # thus, we export the current as best
    print("Congratulations, you brought the network one step closer")
    print("to taking over the world (of backgammon)!!!")
    player1.export_model(filename="nn_best_model")


def self_play():
    """
    Makes a human agent play against another (or the same) human agent.
    """

    player1 = HumanAgent()
    player2 = HumanAgent()

    bg = Backgammon()
    bg.set_player_1(player1)
    bg.set_player_2(player2)
    bg.play()


def random_play():
    """
    Makes a random agent play against another random agent.
    """

    player1 = RandomAgent()
    player2 = RandomAgent()

    bg = Backgammon()
    bg.set_player_1(player1)
    bg.set_player_2(player2)
    bg.play(commentary=True, verbose=True)


def test_play():
    """
    Makes a human agent play against another (or the same) human agent.
    """

    player1 = HumanAgent()
    player2 = get_agent_by_config_name('nn_pg', 'best')

    bg = Backgammon()
    bg.set_player_1(player1)
    bg.set_player_2(player2)
    bg.play()


def main():
    """
    The main function, obviously.
    """
    commands = get_commands_object()

    # Arguments
    args = sys.argv[1:]

    # No argument provided
    if len(args) == 0:
        print("Usage: python3 " + str(sys.argv[0]) + " [COMMAND]")
        print("")
        print("Commands:")
        print("")
        print("    default")
        print("    self-play")
        print("    random-play")
        print("    test-play")
        print("    challange-best-network")
        print("    all_vs_all [competition_test.json or other]")
        print("    jsonhash <path to json>")
        # Stop execution if no argument
        return

    # svo glarb glitchar ekki
    args.append(False)
    if args[0] == "default":
        do_default()
    elif args[0] == "self-play":
        self_play()
    elif args[0] == "random-play":
        random_play()
    elif args[0] == "challange-best-network":
        nn_vs_nn_export_better_player()
    elif args[0] == "test-play":
        test_play()
    elif args[0] == "all_vs_all":
        play_all_vs_all(args[1])
    elif args[0] == "jsonhash":
        try:
            path = " ".join(args[1:])
            print(hash_json(load_file_as_json(path)))
        except:
            print("File is not JSON.")
    else:
        print("Say what?")


if __name__ == "__main__":
    main()
