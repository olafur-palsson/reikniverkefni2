#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: python3 main.py ...
"""
import os, sys
import numpy as np
import agent

def test_1():
    """
    Does something super clever!
    """
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0
    nGames = 1000
    g = 0

    # statistics
    last_100_wins = np.zeros(500)
    highest_win_rate = 0
    output_file_name = agent.get_file_name()

    # play games forever
    while True:
        g = g + 1
        winner = play_a_game(commentary=False)
        winners[str(winner)] += 1

        # reward the agent
        agent.reward_player(winner)

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
            output_result(highest_win_rate, winrate, winners["1"], winners["-1"], g)

    # Default log that was given with code
    print("out of", nGames, "games,")
    print("player", 1, "won", winners["1"],"times and")
    print("player", -1, "won", winners["-1"],"times")



def main():
    """
    The main function, obviously.
    """

    # Arguments
    args = sys.argv[1:]

    if len(args) == 0:
        print("Usage: python3 " + str(sys.argv[0]) + " ...")

    if len(args) == 1 and args[1] == "test":
        test_1()




if __name__ == "__main__":
    main()
