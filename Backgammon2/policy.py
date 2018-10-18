


import numpy as np



class Policy():

    def get_feature_vector(board):
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
        return np.append(features, [jail_features, off_board_features])

    # ! BROKEN, todo: FIX
    def reward_player(reward):
        net.zero_grad()
        y = torch.tensor([reward])
        dont_know_what_im_doing = 0.5
        net.reward(0.5, y)

    def evaluate(board):





