
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


def get_wins(arg_number):
    print(sys.argv[arg_number])
    data = open(sys.argv[arg_number])
    data = filter(lambda line: line.startswith('Player'), data)
    lol = True
    win_history = []
    for line in data:
        numbers = [int(string) for string in line.split() if string.isdigit()]
        wins = numbers[2]
        win_history.append(wins)
    return win_history

def calculate_moving_average(arg_number):
    last_100_wins = np.zeros(100)
    last = 0
    moving_average = np.array([])
    for i ,n in enumerate(get_wins(arg_number)):
        if last == n:
            last_100_wins[i % 100] = -1
        else:
            last_100_wins[i % 100] = 1
        last = n
        moving_average = np.append(moving_average, last_100_wins.sum() / 100)
    return moving_average

filenames = sys.argv[1:]
figure, axis = plt.subplots()
axis.grid()

def plot(arg_number):
    y = calculate_moving_average(arg_number)
    x = np.array([i + 1 for i in range(1000)])
    y = y[-900::5]
    x = x[-900::5]
    label = " ".join(sys.argv[arg_number].split("_"))
    axis.plot(x, y, label=label)

for i in range(len(filenames)):
    plot(i + 1)

axis.legend()
plt.show()
