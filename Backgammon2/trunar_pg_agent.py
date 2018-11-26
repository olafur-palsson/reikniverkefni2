#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trunar_pg_agent.py

A neural network agent.
"""
import numpy as np

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon

from policy_neural_network import PolicyNeuralNetwork

import numpy as np
import torch
from torch.autograd import Variable

import random


device = None

try:
    device = torch.device('cuda') 
except:
    device = torch.device('cpu')



def get_id(n=10):
    return ''.join(random.choice("AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789") for _ in range(n))



class TRUnarPGAgent(AgentInterface):

    def __init__(self, verbose=False, agent_cfg=None, imported=False):
        """
        Creates a neural network agent.

        To load the best NNAgent1 simply set load_best=True

        Args:
            load_best: default `False`
            verbose: default `False`
        """
        AgentInterface.__init__(self)

        self.id = get_id()
        
        # Whether the agent is training.
        self.training = False  

        # Episode that is recorded when playing
        self.episode = []

        # Episodes in memory
        self.episodes = []

        # Discounting factor
        self.gamma = gamma  

        # How many times the player has applied an updated.
        self.iterations = 0
        self.nr_of_updates = 0  # How often a update has been applied.

        # How many games the player has played. (overall)
        self.played_games = 0

        # How many games the player has playeed where he was learning
        self.played_trained_games = 0

        # Whether game is over (False) or active (True)
        self.active = False
        


    def pick_action(self, board, dice, player):
        """

        The state here is `board` and `dice`.  `player` is supposed to be the
        active player.

        Args:
            board (ndarray): backgammon board
            dice (ndarray): a pair of dice
            player: the number for the player on the board who's turn it is.

        Returns:
            A move `move`.
        """

        move = []
        possible_moves, possible_boards = Backgammon.get_all_legal_moves_for_two_dice(board, dice)

        if len(possible_moves) != 0:
            move = self.policy(possible_moves, possible_boards, dice)

        return move

    def add_action(self, action):
        pass

    def export_model(self, filename=False):
        #self.net.export_model(filename=filename)
        pass

    def add_reward(self, reward):
        """
        Adds reward `reward` to this neural network agent.

        NOTE: if you add a reward to the neural network it will immediately
        train.
        """

        # Hence, we only add rewards when we're training..
        #if self.training:
        #    self.pub_stomper.add_reward(reward)
        pass

    def load(self, filename):
        # self.pub_stomper.load(filename)
        pass

    def save(self, save_as_best=False):
        # return self.pub_stomper.save(save_as_best)
        pass


    def policy(self, possible_moves, possible_boards, dice):

        best_move = self.pub_stomper.evaluate(possible_boards)
        move = possible_moves[best_move]

        # gamli kodinn fyrir random
        # move = possible_moves[np.random.randint(len(possible_moves))]
        return move

    def pre_game(self):
        """
        This method is invoked before a game is started.
        """
        pass
    
    def post_game(self):
        """
        This method is invoked after a game has finished.
        """
        
        if self.training:
            self.played_trained_games += 1
        self.played_games += 1



    
    def reset(self):
        self.episode = []
        self.active = False

    def pick_action(self):
        """
        Picks action to take.  Needs to be implemented.
        """

        raise Exception('This method needs to be implemented.')
        
    def pre_state(self):
        """
        This method is invoked before adding a new state.
        """
        
        pass
    
    def post_state(self):
        """
        This method is invoked after adding a new state.
        """
        pass
    
    def verify_state(self, state):
        """
        This method is to verify whether state `state` is legal.
        state: a number
        """
        pass
        
    def add_state(self, state):
        """
        Adds state `state` to episode.
        state: a number
        """
        self.verify_state(state)
        self.pre_state()
        if len(self.episode) % 3 == 0:
            self.episode += [state]
        else:
            raise Exception("Shouldn't have happened!")
        self.post_state()
    
    def pre_action(self):
        """
        This method is invoked before adding an action to the episode list.
        """
        pass
    
    def post_action(self):
        """
        This method is invoked after adding an action to the pisode list.
        """
        pass
    
    def verify_action(self, action):
        """
        This method is used to verify that action `action` is legal.
        action: a number
        """
        pass
    
    def add_action(self, action):
        """
        Adds action `action` to the episode list.
        action: a number
        """
        self.verify_action(action)
        self.pre_action()
        if len(self.episode) % 3 == 1:
            self.episode += [action]
        else:
            raise Exception("Shouldn't have happened!")
        self.post_action()
    
    def pre_reward(self):
        pass
    
    def post_reward(self):
        pass
    
    def verify_reward(self, reward):
        pass
    
    def add_reward(self, reward):
        """
        reward: a number
        """
        self.verify_reward(reward)
        self.pre_reward()
        if len(self.episode) % 3 == 2:
            self.episode += [reward]
        else:
            raise Exception("Shouldn't have happened!")
        self.post_reward()
    
    def pre_game(self):
        """
        This method is invoked before a game is started.
        """
        pass
    
    def post_game(self):
        """
        This method is invoked after a game has finished.
        """
        
        if self.training:
            self.played_trained_games += 1
        self.played_games += 1











# this function is used to find an index to the after-state value table V(s)
def hashit(board):
    base3 = np.matmul(np.power(3, range(0, 9)), board.transpose())
    return int(base3)

# the usual epsilon greedy policy
def epsilongreedy(board, player, epsilon, V, debug = False):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        if debug == True:
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        va[i] = V[hashit(board)]
        board[moves[i]] = 0  # undo move
    return moves[np.argmax(va)]

# this function is used to prepare the raw board as input to the network
# for some games (not here) it may be useful to invert the board and see it from the perspective of "player"
def one_hot_encoding(board, player):
    one_hot = np.zeros( 2 * len(board) )
    one_hot[np.where(board == 1)[0] ] = 1
    one_hot[len(board) + np.where(board == 2)[0] ] = 1
    return one_hot


def softmax_policy(board, player, w1, b1, w2, b2, theta, debug = False):
    """
    Args:
        w1: Variable(torch.randn(m,n, device = device, dtype=torch.float), requires_grad = True)
        b1: Variable(torch.zeros((m,1), device = device, dtype=torch.float), requires_grad = True)
    """
    moves = legal_moves(board)
    na = np.size(moves)
    # feature_vector_boards
    one_hot_boards = np.zeros((18,na))
    for i in range(0, na):
        board[moves[i]] = player #do move
        one_hot_boards[:,i] = one_hot_encoding(board, player) # encode after-state
        board[moves[i]] = 0 # undo move
    # encode the one_hot_boards to create the input
    x = Variable(torch.tensor(one_hot_boards, dtype = torch.float, device = device))
    # now do a forward pass to get the output layer's features
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    pi = torch.mm(theta,h_sigmoid).softmax(1)
    xtheta_mean = torch.sum(torch.mm(h_sigmoid,torch.diagflat(pi)),1)
    xtheta_mean = torch.unsqueeze(xtheta_mean,1)
    m = torch.multinomial(pi, 1)
    
    return moves[m], xtheta_mean

def learnit(numgames, epsilon, lam, alpha, V, alpha1, alpha2, w1, b1, w2, b2, alpha_th, theta):
    gamma = 1 # for completeness
    # play numgames games for training
    for games in range(0, numgames):
        board = np.zeros(9)    # initialize the board (empty)
        # we will use TD(lambda) and so we need to use eligibility traces
        S = [] # no after-state for table V, visited after-states is an empty list
        E = np.array([]) # eligibility traces for table V
        # now we initilize all the eligibility traces for the neural network
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        # player to start is "1" the other player is "2"
        player = 1
        tableplayer = 2
        winner = 0 # this implies a draw
        # start turn playing game, maximum 9 moves
        for move in range(0, 9):
            # use a policy to find action
            if (player == tableplayer): # this one is using the table V
                action = epsilongreedy(np.copy(board), player, epsilon, V)
            else: # this one is using the neural-network to approximate the after-state value
                action, xtheta = softmax_policy(np.copy(board), player, w1, b1, w2, b2, theta)
            # perform move and update board
            board[action] = player
            if (1 == iswin(board, player)): # has this player won?
                winner = player
                break # bail out of inner game loop
            # once both player have performed at least one move we can start doing updates
            if (1 < move):
                if tableplayer == player: # here we have player 1 updating the table V
                    s = hashit(board) # get index to table for this new board
                    delta = 0 + gamma * V[s] - V[sold]
                    E = np.append(E,1) # add trace to this state (note all new states are unique else we would +1)
                    S.append(sold)     # keep track of this state also
                    V[S] = V[S] + delta * alpha * E # the usual tabular TD(lambda) update
                    E = gamma * lam * E
                else: # here we have player 2 updating the neural-network (2 layer feed forward with Sigmoid units)
                    x = Variable(torch.tensor(one_hot_encoding(board, player), dtype = torch.float, device = device)).view(2*9,1)
                    # now do a forward pass to evaluate the new board's after-state value
                    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
                    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
                    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
                    y_sigmoid = y.sigmoid() # squash this with a sigmoid function
                    target = y_sigmoid.detach().cpu().numpy()
                    # lets also do a forward past for the old board, this is the state we will update
                    h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
                    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
                    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
                    y_sigmoid = y.sigmoid() # squash the output
                    delta2 = 0 + gamma * target - y_sigmoid.detach().cpu().numpy() # this is the usual TD error
                    # using autograd and the contructed computational graph in pytorch compute all gradients
                    y_sigmoid.backward()
                    # update the eligibility traces using the gradients
                    Z_w2 = gamma * lam * Z_w2 + w2.grad.data
                    Z_b2 = gamma * lam * Z_b2 + b2.grad.data
                    Z_w1 = gamma * lam * Z_w1 + w1.grad.data
                    Z_b1 = gamma * lam * Z_b1 + b1.grad.data
                    # zero the gradients
                    w2.grad.data.zero_()
                    b2.grad.data.zero_()
                    w1.grad.data.zero_()
                    b1.grad.data.zero_()
                    # perform now the update for the weights
                    delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
                    w1.data = w1.data + alpha1 * delta2 * Z_w1
                    b1.data = b1.data + alpha1 * delta2 * Z_b1
                    w2.data = w2.data + alpha2 * delta2 * Z_w2
                    b2.data = b2.data + alpha2 * delta2 * Z_b2
                    # now perform the update for the Actor
                    grad_ln_pi = h_sigmoid - xtheta
                    theta.data = theta.data + alpha_th*delta2*grad_ln_pi.view(1,len(grad_ln_pi))
 

            # we need to keep track of the last board state visited by the players
            if tableplayer == player:
                sold = hashit(board)
            else:
                xold = Variable(torch.tensor(one_hot_encoding(board, player), dtype=torch.float, device = device)).view(2*9,1)
            # swap players
            player = getotherplayer(player)

        # The game epsiode has ended and we know the outcome of the game, and can find the terminal rewards
        if winner == tableplayer:
            reward = 0
        elif winner == getotherplayer(tableplayer):
            reward = 1
        else:
            reward = 0.5
        # Now we perform the final update (terminal after-state value is zero)
        # these are basically the same updates as in the inner loop but for the final-after-states (sold and xold)
        # first for the table (note if reward is 0 this player actually won!):
        delta = (1.0 - reward) + gamma * 0 - V[sold]
        E = np.append(E,1) # add one to the trace (recall unique states)
        S.append(sold)
        V[S] = V[S] + delta * alpha * E
        # and then for the neural network:
        h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        delta2 = reward + gamma * 0 - y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces
        Z_w2 = gamma * lam * Z_w2 + w2.grad.data
        Z_b2 = gamma * lam * Z_b2 + b2.grad.data
        Z_w1 = gamma * lam * Z_w1 + w1.grad.data
        Z_b1 = gamma * lam * Z_b1 + b1.grad.data
        # zero the gradients
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta2 * Z_w1
        b1.data = b1.data + alpha1 * delta2 * Z_b1
        w2.data = w2.data + alpha2 * delta2 * Z_w2
        b2.data = b2.data + alpha2 * delta2 * Z_b2
        # now perform the update for the Actor
        grad_ln_pi = h_sigmoid - xtheta
        theta.data = theta.data + alpha_th*delta2*grad_ln_pi.view(1,len(grad_ln_pi))





def competition(V, w1, b1, w2, b2, theta, epsilon = 0.0, debug = False):
    board = np.zeros(9)          # initialize the board
    # player to start is "1" the other player is "2"
    player = 1
    tableplayer = 2
    winner = 0 # default draw
    # start turn playing game, maximum 9 moves
    for move in range(0, 9):
        # use a policy to find action, switch off exploration
        if (tableplayer == player):
            action = epsilongreedy(np.copy(board), player, epsilon, V, debug)
        else:
            action, _ = softmax_policy(np.copy(board), player, w1, b1, w2, b2, theta)
        # perform move and update board (for other player)
        board[action] = player
        if debug: # print the board, when in debug mode
            symbols = np.array([" ", "X", "O"])
            print("player ", symbols[player], ", move number ", move+1, ":", action)
            print(symbols[board.astype(int)].reshape(3,3))

        if (1 == iswin(board, player)): # has this player won?
            winner = player
            break
        player = getotherplayer(player) # swap players
    return winner








# global after-state value function, note this table is too big and contrains states that
# will never be used, also each state is unique to the player (no one after-state seen by both players)
V = np.zeros(hashit(2 * np.ones(9)))

alpha = 0.1 # step size for tabular learning
alpha1 = 0.1 # step sizes using for the neural network (first layer)
alpha2 = 0.1 # (second layer), the critic's neural network
alpha_th = 0.0001 # for the Actor's thetas, uses softmax policy
epsilon = 0.1 # exploration parameter used by epsilon greedy player (table)
lam = 0.4 # lambda parameter in TD(lam-bda)

# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(9*9,2*9, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((9*9,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1,9*9, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

# here I have added the linear weights for the Actor
theta = 0.01*torch.ones((1,9*9), device = device, dtype=torch.float)

# now perform the actual training and display the computation time
import time
start = time.time()
training_steps = 1
learnit(training_steps, epsilon, lam, alpha, V, alpha1, alpha2, w1, b2, w2, b2, alpha_th, theta)
end = time.time()
print(end - start)


wins_for_player_1 = 0
draw_for_players = 0
loss_for_player_1 = 0
competition_games = 100
for j in range(competition_games):
    winner = competition(V, w1, b1, w2, b2, theta, epsilon, debug = False)
    if (winner == 1):
        wins_for_player_1 += 1.0
    elif (winner == 0):
        draw_for_players += 1.0
    else:
        loss_for_player_1 += 1.0

print(wins_for_player_1, draw_for_players, loss_for_player_1)
# lets also play one deterministic games:
winner = competition(V, w1, b1, w2, b2, theta, 0, debug = True)








