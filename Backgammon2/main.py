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


# Print results out to a file (every 100 games)
def output_result(agent, highest_win_rate, win_rate, p1wins, p2wins, games_played):
    """
    Save something from `do_default()`.
    """
    file_name = "results/" + agent.get_file_name() + "_result.pt"
    Path(file_name).touch()
    file = open(file_name, "w")
    file.write("Highest win rate last 500: " + str(highest_win_rate) + "\n")
    file.write("End win rate: " +  str(win_rate) + "\n")
    file.write("Wins: " + str(p1wins) + "\n")
    file.write("Loses: " + str(p2wins) + "\n")
    file.write("Games played: " + str(games_played) + "\n")
    file.close()


def do_default():
    """
    Do the default think as in Backgammon.py.
    """
    
    winners = {
        "1": 0,
        "-1": 0
    }

    nGames = 1000
    g = 0

    # statistics
    last_100_wins = np.zeros(500)
    highest_win_rate = 0

    player1 = NNAgent1()
    player2 = RandomAgent()

    output_file_name = player1.get_file_name()

    # play games forever
    while True:
        g = g + 1

        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        winner = bg.play()

        winners[str(winner)] += 1

        # Reward the neural network agent
        player1.reward_player(winner)

        # Gather the win/loss of last 500 games
        win = winner if winner > 0 else 0
        last_100_wins[g % 500] = win
        winrate = np.sum(last_100_wins) / 5
        highest_win_rate = winrate if winrate > highest_win_rate else highest_win_rate

        # Print out a log of game-stats
        print("")
        print(winner)
        print("Player 1 : Player 2 : Total     " + str( winners["1"]) + " : " + str(winners["-1"]) +  " : " + str(g) +  "        moving average 500:   " +  str(winrate) +  "%" + " (max - stddev =" + str(highest_win_rate - 2) + "%), std-dev of this is ~2%")
        print("")
        print("")
        print("")
        print("")
        if g % 10 == 0:
            output_result(player1, highest_win_rate, winrate, winners["1"], winners["-1"], g)

    # Default log that was given with code
    print("out of", nGames, "games,")
    print("player", 1, "won", winners["1"],"times and")
    print("player", -1, "won", winners["-1"],"times")



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


def main():
    """
    The main function, obviously.
    """

    commands = get_commands_object()

    # Arguments
    args = sys.argv[1:]

    if len(args) == 0:
        print("Usage: python3 " + str(sys.argv[0]) + " [COMMAND]")
        print("")
        print("Commands:")
        print("")
        print("    default")
        print("    self-play")
        print("    random-play")

    if len(args) == 1 and args[0] == "default":
        do_default()
    elif len(args) == 1 and args[0] == "self-play":
        self_play()
    elif len(args) == 1 and args[0] == "random-play":
        random_play()





if __name__ == "__main__":
    main()
