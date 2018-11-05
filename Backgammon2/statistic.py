from pathlib import Path
import numpy as np


class Statistic():
    # notum til ad skoda hvort tauganetid er consistently betra

    goal_win_rate = 0.54

    last_5000_wins = np.zeros(1500)
    last_500_wins = np.zeros(500)
    winners = [0, 0]
    games_played = 0
    highest_win_rate = 0
    win_rate = 0
    verbose = False

    def __init__(self, agent=None, verbose=False):

        if agent is None:
            raise Exception("lawl")

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