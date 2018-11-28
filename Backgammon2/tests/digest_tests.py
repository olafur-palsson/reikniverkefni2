
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_wins():
    data = open(sys.argv[1])
    data = filter(lambda line: line.startswith('Player'), data)
    lol = True
    win_history = []
    for line in data:
        numbers = [int(string) for string in line.split() if string.isdigit()]
        wins = numbers[2]
        win_history.append(wins)
    return win_history


def calculate_moving_average():
    last_100_wins = np.zeros(100)
    last = 0
    moving_average = []
    for i ,n in enumerate(get_wins()):
        if last == n:
            last_100_wins[i % 100] = -1
        else:
            last_100_wins[i % 100] = 1
        last = n
        moving_average.append(last_100_wins.sum() / 100)

    return moving_average


# calculate data for axis of graph
y = calculate_moving_average()
x = np.array([i + 1 for i in range(1000)])

# cut off first 100 games (because moving average of last 100)
y = y[-900:]
x = x[-900:]

figure, axis = plt.subplots()
axis.grid()
axis.plot(x, y)

print(y)
print(x)

plt.show()
