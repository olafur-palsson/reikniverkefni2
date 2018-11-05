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
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.nn_agent_1 import NNAgent1
from agents.nn_agent_best_so_far import BestNNAgent

from statistic import Statistic

from glarb import do_glarb






# Set logs
verbose = True




def do_default():
    """
    Play with a neural network against random
    """

    player1 = NNAgent1(verbose=True)
    player2 = RandomAgent()

    stats = Statistic(player1, verbose=True)

    # play games forever
    while True:

        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        winner = bg.play()

        # Reward the neural network agent
        player1.reward_player(winner)

        stats.add_win(winner)

        # Print out a log of game-stats
        if stats.games_played % 10 == 0:
            stats.output_result()

def nn_vs_nn_export_better_player():
    player1 = NNAgent1(verbose=True)
    player2 = BestNNAgent()

    stats = Statistic(player1, verbose=True)

    while True:
        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        winner = bg.play()

        player1.reward_player(winner)
        player2.reward_player(-1 * winner)

        stats.add_win(winner)

        if stats.nn_is_better():
            break

    # only way to reach this point is if the current
    # neural network is better than the BestNNAgent()
    # ... at least I think so
    # thus, we export the current as best
    print("Congratulations, you brought the network one step closer")
    print("to taking over the world (of backgammon)!!!")
    player1.export_model(file_name="nn_best_model")


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
    player2 = BestNNAgent()

    bg = Backgammon()
    bg.set_player_1(player1)
    bg.set_player_2(player2)
    bg.play()





def test_glarb():
    do_glarb()




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
        print("    challenge-best-network")
        # Stop execution if no argument
        return

    if args[0] == "default":
        do_default()
    elif args[0] == "self-play":
        self_play()
    elif args[0] == "random-play":
        random_play()
    elif args[0] == "challenge-best-network":
        nn_vs_nn_export_better_player()
    elif args[0] == "test-play":
        test_play()
    elif args[0] == "glarb":
        test_glarb()
    else:
        print("Say what?")





if __name__ == "__main__":
    main()
