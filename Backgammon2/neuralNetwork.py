
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

# Make this one the same as the output of the featuere vector
input_width, output_width = 464, 1

# Decide how many hidden layers
hidden_layers_width = [250, 250, output_width]

# If update is with with n-step algorithm, n = td_n
temporal_delay = 3

counter = 0
default_file_name = "_".join(str(datetime.datetime.now()).split(" "))

# make this one output [nn.Linear, nn.Linear...] or whatever layers you would like, then the rest is automatic
def make_layers():
    layers = []
    last_width = input_width
    for width in hidden_layers_width:
        layers.append(nn.Linear(last_width, width))
        last_width = width
        # layers.append(nn.ReLU()) # uncomment for ReLU
        # layers.append(nn.Dropout(p=0.025)) # uncomment for drop-out

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

    def make_file_name_from_string(self, file_name_root_string):
        # sets class-wide filename for exporting to files
        self.model_file_name = "./tests/" + file_name_root_string + " model.pt"
        self.optimizer_file_name = "./tests/" + file_name_root_string + " optim.pt"
        self.settings_file_name = "results/" + file_name_root_string + "_settings.pt"


    def __init__(self, load_file_name=False, export=False):
        # set up file_names for exporting
        self.file_name = load_file_name if load_file_name else default_file_name
        self.make_file_name_from_string(self.file_name)

        # make layers in neural network and make the network sequential
        # (i.e) input -> layer_1 -> ... -> layer_n -> output  for layers in 'make_layers()'
        self.model = nn.Sequential(*make_layers())

        # initialize prediction storage
        self.predictions = torch.empty((1), dtype = dtype, requires_grad=True)

        # set loss function for backprop (usage is optional)
        self.loss_fn = loss_fn = torch.nn.MSELoss(size_average=False)

        # set optimizer for adjusting the weights (e.g Stochastic Gradient Descent, SGD)
        # Note: learning_rate is at the top of the script
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=learning_rate)

        # True if should export, False if we throw it away after running
        self.export = export

        # Game counter for intervals
        self.counter = 0

        # Reward storage for batched learning
        self.rewards = []

        # If we want to load a model we input the name of the file, if exists -> load
        if load_file_name:
            # import model
            self.optimizer.load_state_dict(torch.load("./tests/" + self.file_name + " optim.pt"))
            self.model.load_state_dict(torch.load("./tests/" + self.file_name + " model.pt"))
        else:
            # export current settings
            self.make_settings_file()

    def export_model(self):
        torch.save(self.model.state_dict(), self.model_file_name)
        torch.save(self.optimizer.state_dict(), self.optimizer_file_name)

    # run a feature vector through the model accumulating greadient
    def run_decision(self, board_features):
        vector = board_features
        prediction = self.model(board_features)
        self.predictions = torch.cat((self.predictions, prediction.double()))

    # run a feature vector through the model without accumulating gradient
    def predict(self, board_features):
        with torch.no_grad():
            return self.model(board_features)

    # Function run on the end of each game.
    def get_reward(self, reward):
        """
            We at this point have accumulated predictions of the network in self.predictions
            Here we decide what values we should we should move towards. We shall name that
            vector 'y'
        """

        episode_length = len(self.predictions)
        y = torch.ones((episode_length), dtype=dtype, requires_grad=False) * reward

        # TD valued reward
        with torch.no_grad():
            for i in range(len(self.predictions)):
                if i == len(self.predictions) - temporal_delay:
                    break
                y[i] = self.predictions[i + temporal_delay]

        self.rewards.append(y)

        # Sum of squared error as loss
        loss = (self.predictions - y).pow(2).sum()
        # Zero all accumulated gradients
        self.optimizer.zero_grad()
        # Recalculate gradients based on 'loss' (i.e. what it takes for loss -> 0)
        loss.backward()
        # Use optimizer to calculate new weights
        self.optimizer.step()

        # Export model each 100 episodes
        if counter % 100 == 0 and self.export:
            self.export_model()

        # Log out statistics of current game
        print("First state td value")
        print(y[0])
        print("Prediction of last state ('-' means guessed wrong, number is confidence, optimal = 1 > p > 0.8) ")
        print(str(float(self.predictions[episode_length - 1] * reward)))
        print("First state")
        print(str(float(self.predictions[0])))

        # reset empty predictions
        self.predictions = torch.empty(0, dtype = dtype, requires_grad=True)
        # kalla a predictions.sum til ad kalla bara einu sinni a
        # loss.backward()
