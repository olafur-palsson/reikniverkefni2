


import numpy as np
import torch



class Policy():
    def get_feature_vector(self, board):
        return self.get_raw_data(board)
        main_board = board[1:25]
        jail1, jail2, off1, off2 = board[25], board[26], board[27], board[28]
        features = np.array([])

        # naum i feature vector af adalsvaedinu
        for position in main_board:
            vector = np.zeros(4)
            sign = -1 if position < 0 else 1
            for i in range(int(abs(position))):
                if i > 3:
                    vector[3] = sign * (abs(position) - 3) / 2
                    break
                vector[i] = position/abs(position)
            features = np.append(features, vector)

        # jail feature-ar
        jail_features = np.array([jail1, jail2]) * 0.5

        # features fyrir hversu margir eru borne off
        off_board_features = np.array([off1, off2]) * (0.066667)
        bias_vector = np.array([1, 1])
        features =  np.append(features, [jail_features, off_board_features, bias_vector])
        features = torch.from_numpy(features).float()
        features.requires_grad = True
        return features

    def get_raw_data(self, board):
        features = np.array([])
        for position in board:
            vector = np.zeros(16)
            for i in range(int(position)):
                vector[i] = 1
            features = np.append(features, vector)
        features = torch.from_numpy(features).float()
        features.requires_grad = True
        return features


    # ! BROKEN, todo: FIX
    def get_reward(self, reward):
        print("Reward function not set")

    def evaluate(self, board):
        print("Evaluation function not set")
