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


# Set logs
verbose = True


class Statstics():
    # notum til ad skoda hvort tauganetid er consistently betra

    # likur a ad nn se betra eru tha 98.9% i hvert skipti sem er tekkad
    # vid tekkum hver 100 skipti svo vid ovart setjum ekki jafn gott/verra net
    # i stadinn fyrir thad besta (ef thad winnur 51% skipta tha verdur thad
    # nogu heppid a ~2000 leikja fresti)
    goal_win_rate = 0.537


    last_5000_wins = np.zeros(1000)
    last_500_wins = np.zeros(500)
    winners = [0, 0]
    games_played = 0
    highest_win_rate = 0
    win_rate = 0
    verbose = False

    def __init__(self, agent, verbose=False):
        self.agent = agent
        if verbose:
            self.verbose = True

    def two_digits(self, double_number):
        return "{0:.2f}".format(double_number)

    def update_win_rate(self, winner):
        win = 1 if winner > 0 else 0
        self.last_500_wins[self.games_played % 500] = win
        self.last_5000_wins[self.games_played % 5000] = win
        self.win_rate = np.sum(self.last_500_wins) / 5
        if self.win_rate > self.highest_win_rate:
            self.highest_win_rate = self.win_rate

    def nn_is_better(self):
        if np.sum(self.last_5000_wins) / 5000 > self.goal_win_rate:
            return True
        return False

    def add_win(self, winner, verbose=False):
        self.games_played += 1
        self.update_win_rate(winner)
        i = 0 if winner == 1 else 1
        self. winners[i] += 1
        if self.verbose:
            self.verbose_print()

    def verbose_print(self):
        string =      "Player 1 : Player 2 : Total     "
        string +=     str(self.winners[0]) + " : " + str(self.winners[1]) + " : " + str(self.games_played)
        string +=     "        moving average 500:   "
        string +=     str(self.win_rate) + "%"
        string +=     " (max - stddev = "
        string +=     str(self.two_digits(self.highest_win_rate - 2)) + "%), std-dev of this is ~2%"
        print("")
        print(string)
        print("")
        print("")
        print("")
        print("")

    # Print results out to a file (every 100 games)
    # agent object needs to have a get_file_name() method!
    def output_result(self):
        """
        Save something from `do_default()`.
        """
        file_name = "results/" + self.agent.get_file_name() + "_result.pt"
        Path(file_name).touch()
        file = open(file_name, "w")
        file.write("Highest win rate last 500: " + str(self.highest_win_rate) + "\n")
        file.write("End win rate: " +  str(self.win_rate) + "\n")
        file.write("Wins: " + str(self.winners[0]) + "\n")
        file.write("Loses: " + str(self.winners[1]) + "\n")
        file.write("Games played: " + str(self.games_played) + "\n")
        file.close()

def do_default():
    """
    Play with a neural network against random
    """

    player1 = NNAgent1(verbose=True)
    player2 = RandomAgent()

    stats = Statstics(player1, verbose=True)

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

    stats = Statstics(player1, verbose=True)

    while True:
        bg = Backgammon()
        bg.set_player_1(player1)
        bg.set_player_2(player2)
        winner = bg.play()

        player1.reward_player(winner)
        player2.reward_player(-1 * winner)

        stats.add_win(winner)

        if stats.nn_is_better() and stats.games_played % 100 == 0:
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
        print("    challange-best-network")
        # Stop execution if no argument
        return

    if args[0] == "default":
        do_default()
    elif args[0] == "self-play":
        self_play()
    elif args[0] == "random-play":
        random_play()
    elif args[0] == "challange-best-network":
        nn_vs_nn_export_better_player()

    print("Invalid parameter")





if __name__ == "__main__":
    main()
