
import torch
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as Optimizer
import numpy as np
import datetime
from functools import reduce
from torch.autograd import Variable
from pathlib import Path

learning_rate = 5e-5
dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

input_width, output_width = 464, 1
hidden_layers_width = [700, 700, 700, output_width]
td_n = 3
# hidden_layers_width = [150, 150]

all_width = 70
counter = 0
default_file_name = "".join(str(datetime.datetime.now()).split(" "))

def make_layers():
    layers = []

    last_width = input_width

    """
    layers.append(nn.Linear(last_width, all_width))
    last_width = all_width
    for i in range(20):
        layers.append(nn.Linear(all_width, all_width))

    """

    for width in hidden_layers_width:
        layers.append(nn.Linear(last_width, width))
        last_width = width
        # layers.append(nn.ReLU()) # uncomment for ReLU
        # layers.append(nn.Dropout(p=0.025))


    final = nn.Linear(last_width, output_width)
    # layers.append(nn.ReLU()) # uncomment for ReLU
    layers.append(final)
    return layers


class BasicNetworkForTesting():


    def make_settings_file(self):
        Path(self.settings_file_name).touch()
        file = open(self.settings_file_name, "w")
        file.write("Input vector size: " + str(input_width) + "\n")
        file.write("Hidden layers: " + str(hidden_layers_width) + "\n")
        file.write("Learning rate: " + str(learning_rate) + "\n")
        file.close()

    def __init__(self, load_file_name=False, export=False):
        self.model = nn.Sequential(*make_layers())
        self.predictions = torch.empty((1), dtype = dtype, requires_grad=True)
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=learning_rate)
        self.export = export
        self.file_name = load_file_name if load_file_name else default_file_name
        self.model_file_name = "./tests/" + self.file_name + " model.pt"
        self.optimizer_file_name = "./tests/" + self.file_name + " optim.pt"
        self.settings_file_name = "results/" + self.file_name + "_settings.pt"
        self.counter = 0
        self.rewards = []
        if load_file_name:
            self.optimizer.load_state_dict(torch.load("./tests/" + self.file_name + " optim.pt"))
            self.model.load_state_dict(torch.load("./tests/" + self.file_name + " model.pt"))
        else:
            self.make_settings_file()

    def export_model(self):
        torch.save(self.model.state_dict(), self.model_file_name)
        torch.save(self.optimizer.state_dict(), self.optimizer_file_name)

    def run_decision(self, board_features):
        vector = board_features
        prediction = self.model(board_features)
        self.predictions = torch.cat((self.predictions, prediction.double()))

    def predict(self, board_features):
        with torch.no_grad():
            return self.model(board_features)

    def get_reward(self, reward, exp_return):
        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype, requires_grad=False) * reward

        with torch.no_grad():
            for i in range(len(self.predictions)):
                if i == len(self.predictions) - td_n:
                    break
                y[i] = self.predictions[i + td_n]

        self.rewards = append(y)

        """
        exp_return = 0 # thessi lina laetur y[i] = reward * i
        # lata early moves fa expected return med sma nudge, late moves fa meira reward, a milli er progressive
        # y[seinast] = reward
        # y[0] er u.th.b. exp_return

        for i in range(episode_length):
          y[i] = (y[i] * i + (episode_length - (i + 1) ) * exp_return) / (episode_length - 1)
        """

        length = 0
        for reward_vector in self.rewards:
            length = len(reward)

        if counter % 20 == 0:


        loss = (self.predictions - y).pow(2).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.predictions - torch.ones((episode_length), dtype=dtype) * torch.mean(self.predictions))
        if counter % 100 == 0 and self.export:
            self.export_model()

        print("First state td value")
        print(y[0])
        print("Prediction of last state ('-' means guessed wrong, number is confidence, optimal = 1 > p > 0.8) ")
        print(str(float(self.predictions[episode_length - 1] * reward)))
        print("First state")
        print(str(float(self.predictions[0])))
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()
