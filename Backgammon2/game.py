#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class implementation of backgammon.

Things to keep in mind:

"""
import numpy as np
import agent
from pathlib import Path


def str_symbol(board, pos, height):
    """
    A helper class for drawing the string representation for `Backgammon`.

    
    """

    #
    # height = 2
    # height = 1
    # height = 0

    if abs(board[pos]) > height:
        if board[pos] > 0:
            # White
            return "x"
        else:
            # Black
            return "o"
    else:
        return " "


class Backgammon:
    """
    Implementation of Backgammon.

    This implementation makes no assumption about the player, i.e. whether
    the player is a computer controlled bot or a human player.
    """


    def __init__(self):
        """
        Notice: when the player is asked for an `action` that player always
        sees him-/herself as player nr. 1

        NOTE: I'm not sure about that, go over it.

        Player 1 has the number `1` (positive) and is denoted as `x`
        
        Player 2 has the number `-1` (negative) and is denoted as `o`.
        """
        self.board = np.zeros(0)  # Just to get type information

        self.player_1 = None  # TODO: make instance of future interface
        self.player_2 = None  # TODO: make instance of future interface

        # `1` for player 1 and `-1` for player 2
        self.active_player = 0  # Just to get type information

        self.reset()


    def set_player_1(self, player):
        """
        Sets player 1 to `player`.
        """
        self.player_1 = player


    def set_player_2(self, player):
        """
        Sets player 2 to `player`.
        """
        self.player_2 = player


    def reset(self):
        """
        Initializes the board and resets everything
        to default.
        """
        # Create board and initialize it
        
        #
        # Meaning of array.
        #
        # o: -
        # x: +
        #
        #    <-- x --                                       -- o -->    
        # ? .-------------.-------------.-------------.-------------.
        # ? |           x |           o | x           | o           | J J O O        
        # ? |           x |           o | x           | o           | 1 2 1 2                 
        # ? |           x |   x       o | x       o   | o           |                         
        # ? | o         x |   x       o | x       o   | o         x | J J O O                
        # ? | o         x |   x       o | x       o   | o         x | 1 2 1 2  
        # ? '-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'
        #                         1 1 1   1 1 1 1 1 1   1 2 2 2 2 2   2 2 2 2
        # 0   1 2 3 4 5 6   7 8 9 0 1 2   3 4 5 6 7 8   9 0 1 2 3 4   5 6 7 8 
        #
        # Where ? means I don't know what that means.
        #
        # J1 is jail 1
        # J2 is jail 2
        #
        # O1 is off 1
        # O2 is off 2
        #
        #
        #

        self.board = np.zeros(29)
        self.board[1] = -2
        self.board[6] = 5
        self.board[8] = 3
        self.board[12] = -5
        self.board[13] = 5
        self.board[17] = -3
        self.board[19] = -5
        self.board[24] = 2

        # Reset active player
        # `1` for player 1 and `-1` for player 2
        self.active_player = 0  # Just to get type information


    def roll_dice(self):
        """
        Rolls the dice
        """
        dice = np.random.randint(1,7,2)
        return dice


    def is_game_over(self):
        """
        Returns `True` if the game is over, otherwise `False`.
        """
        return self.board[27] == 15 or self.board[28] == -15


    def get_winner(self):
        """
        Returns -1 or 1 if either player has won, 0 if no one has won.
        """
        # TODO: implement
        raise Exception("IMPLEMENT")


    def __verify(self):
        """
        Checks this game for obvious errors.
        
        Raises an exception if something goes wrong.
        """
        
        if sum(self.board[self.board > 0]) != 15 or sum(self.board[self.board < 0]) != -15:
            raise Exception("Too many pieces on the board.")


    def pretty_print(self, board = None):
        """
        Prints the board.
        """
        
        if board is None:
            board = self.board

        string = str(np.array2string(board[1:13])+'\n'+
                     np.array2string(board[24:12:-1])+'\n'+
                     np.array2string(board[25:29]))
        print(string)


    def get_flipped_board(self):
        """
        Returns the flipped board.
        """

        # alias
        board = self.board

        board = board * (-1)
        main_board = board[24:0:-1]
        jail1, jail2, off1, off2 = board[26], board[25], board[28], board[27]
        main_with_zero = np.insert(main_board, 0, board[0])
        new_board = np.append(main_with_zero, np.array([jail1, jail2, off1, off2]))

        return new_board


    def flip_board(self):
        """
        Flips the board.

        NOTE: this flip is in-place
        """

        self.board = self.get_flipped_board()
    

    def clone(self):
        """
        Creates a clone of this backgammon game.
        """
        pass


    def copy_board(self):
        """
        Returns a copy of the board of this game in its current state.
        """
        return np.copy(self.board)


    def play(self, commentary = True):
        """
        Returns which player won (1 for player 1, -1 for player 2).
        """
        
        # Initialize game
        self.reset()

        # player 1 starts
        #
        # TODO: roll a dice in future to determine which player stars, i.e.
        # the player which rolls higher starts, and uses those dice in his/her
        # first turn.

        self.active_player = 1
        
        #  1 -> player 1
        # -1 -> player 2

        # Keep making turns until one or the other player wins.
        while not self.is_game_over():

            if commentary:
                print("Lets go player: ", self.active_player)

            # Roll dice
            dice = self.roll_dice()

            # 
            turns = 1
            if dice[0] == dice[1]:
                turns = 2

            # Make a move (2 moves if the same number appears on the dice)
            for i in range(turns):
                board_copy = self.copy_board()

                move = None

                if self.active_player == 1:
                    move = self.player_1.action(board_copy, dice, 1, i)
                elif self.active_player == -1:
                    move = self.player_2.action(board_copy, dice, 1, i)
                else:
                    raise Exception("This shouldn't have happened.")

                # update the board
                if len(move) != 0:
                    for m in move:
                        board = update_board(board, m)

                if commentary:
                    print("move from player",player,":")
                    print("board:")
                    if self.active_player == 1:
                        self.pretty_print()
                    else:
                        self.pretty_print(self.get_flipped_board())

            # players take turns
            self.active_player = -self.active_player
            self.flip_board()

        return -self.active_player

    def __str__(self):
        """
        If you evaluate `str(x)` where `x` is an instance of `Backgammon` then
        this string (which is constructed here) is returned.
        """

        # TODO: finish this

        board = [
            0,  # White's jail (white moves up)
            0,  # Black's winning thing
            
            +2,  0,  0,  0,  0, -5,  # Black's home board
             0, -3,  0,  0,  0, +5,

            -5,  0,  0,  0, +3,  0,
            +5,  0,  0,  0,  0, -2,  # White's home board

            0, # White's winning thing
            0  # Black's jail  (black moves down)
        ]

        original_board = self.board

        # Map the board of this class to a board this function can represent.


        board = self.board
        msg = ""
        height = np.amax([abs(x) for x in board])
        dir_banner = "           <-- o --                                       -- x -->"
        top_banner = "  .---.---.-------------.-------------.-------------.-------------.---.---."
        bottom_banner = "  '-^-'-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-'-^-'"
        num_banner_1 = "                              1 1 1 1   1 1 1 1 1 1   2 2 2 2 2 2   2   2"
        num_banner_2 = "    0   1   2 3 4 5 6 7   8 9 0 1 2 3   4 5 6 7 8 9   0 1 2 3 4 5   6   7"
        msg += dir_banner
        msg += "\n"
        msg += top_banner
        msg += "\n"
        for i in range(height):
            submsg = ""
            j = height - i
            h = j - 1

            if height > 10 and i < 10:
                submsg += "0" + str(j)
            else:
                submsg += str(j) 
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 0, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 1, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 2, h)
            submsg += " "
            submsg += str_symbol(board, 3, h)
            submsg += " "
            submsg += str_symbol(board, 4, h)
            submsg += " "
            submsg += str_symbol(board, 5, h)
            submsg += " "
            submsg += str_symbol(board, 6, h)
            submsg += " "
            submsg += str_symbol(board, 7, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 8, h)
            submsg += " "
            submsg += str_symbol(board, 9, h)
            submsg += " "
            submsg += str_symbol(board, 10, h)
            submsg += " "
            submsg += str_symbol(board, 11, h)
            submsg += " "
            submsg += str_symbol(board, 12, h)
            submsg += " "
            submsg += str_symbol(board, 13, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 14, h)
            submsg += " "
            submsg += str_symbol(board, 15, h)
            submsg += " "
            submsg += str_symbol(board, 16, h)
            submsg += " "
            submsg += str_symbol(board, 17, h)
            submsg += " "
            submsg += str_symbol(board, 18, h)
            submsg += " "
            submsg += str_symbol(board, 19, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 20, h)
            submsg += " "
            submsg += str_symbol(board, 21, h)
            submsg += " "
            submsg += str_symbol(board, 22, h)
            submsg += " "
            submsg += str_symbol(board, 23, h)
            submsg += " "
            submsg += str_symbol(board, 24, h)
            submsg += " "
            submsg += str_symbol(board, 25, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 26, h)
            submsg += " "
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 27, h)
            submsg += " "
            submsg += "|"
            submsg += "\n"
            msg += submsg
        msg += bottom_banner
        msg += "\n"
        msg += num_banner_1
        msg += "\n"
        msg += num_banner_2
        msg += "\n"
        msg += "\n"
        msg += "    Turn: x"
        msg += "\n"
        return msg


def legal_move(board, die):
    # finds legal moves for a board and one dice
    # inputs are some BG-board, the number on the die and which player is up
    # outputs all the moves (just for the one die)
    possible_moves = []

    if board[25] > 0:
        start_pip = 25-die
        if board[start_pip] > -2:
            possible_moves.append(np.array([25,start_pip]))

    # no dead pieces
    else:
        # adding options if player is bearing off
        if sum(board[7:25]>0) == 0:
            if (board[die] > 0):
                possible_moves.append(np.array([die,27]))

            elif not game_over(board): # smá fix
                # everybody's past the dice throw?
                s = np.max(np.where(board[1:7]>0)[0]+1)
                if s<die:
                    possible_moves.append(np.array([s,27]))

        possible_start_pips = np.where(board[0:25]>0)[0]

        # finding all other legal options
        for s in possible_start_pips:
            end_pip = s-die
            if end_pip > 0:
                if board[end_pip] > -2:
                    possible_moves.append(np.array([s,end_pip]))

    return possible_moves

def legal_moves(board, dice):
    # finds all possible moves and the possible board after-states
    # inputs are the BG-board, the dices rolled and which player is up
    # outputs the possible pair of moves (if they exists) and their after-states

    moves = []
    boards = []

    # try using the first dice, then the second dice
    possible_first_moves = legal_move(board, dice[0])
    for m1 in possible_first_moves:
        temp_board = update_board(board,m1)
        possible_second_moves = legal_move(temp_board,dice[1])
        for m2 in possible_second_moves:
            moves.append(np.array([m1,m2]))
            boards.append(update_board(temp_board,m2))

    if dice[0] != dice[1]:
        # try using the second dice, then the first one
        possible_first_moves = legal_move(board, dice[1])
        for m1 in possible_first_moves:
            temp_board = update_board(board,m1)
            possible_second_moves = legal_move(temp_board,dice[0])
            for m2 in possible_second_moves:
                moves.append(np.array([m1,m2]))
                boards.append(update_board(temp_board,m2))

    # if there's no pair of moves available, allow one move:
    if len(moves)==0:
        # first dice:
        possible_first_moves = legal_move(board, dice[0])
        for m in possible_first_moves:
            moves.append(np.array([m]))
            boards.append(update_board(temp_board,m))

        # second dice:
        possible_first_moves = legal_move(board, dice[1])
        for m in possible_first_moves:
            moves.append(np.array([m]))
            boards.append(update_board(temp_board,m))

    return moves, boards

def update_board(board, move):
    # updates the board
    # inputs are some board, one move and the player
    # outputs the updated board
    board_to_update = np.copy(board)

    # if the move is there
    if len(move) > 0:
        startPip = move[0]
        endPip = move[1]

        # moving the dead piece if the move kills a piece
        kill = board_to_update[endPip]==(-1)
        if kill:
            board_to_update[endPip] = 0
            jail = 26
            board_to_update[jail] = board_to_update[jail] - 1

        board_to_update[startPip] = board_to_update[startPip]-1
        board_to_update[endPip] = board_to_update[endPip]+1

    return board_to_update


def random_agent(board_copy,dice,i):
    # random agent
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move randomly

    # check out the legal moves available for dice throw
    possible_moves, possible_boards = legal_moves(board_copy, dice)

    if len(possible_moves) == 0:
        return []
    else:
        move = possible_moves[np.random.randint(len(possible_moves))]
    return move



# Print results out to a file (every 100 games)
def output_result(highest_win_rate, win_rate, p1wins, p2wins, games_played):
    file_name = "results/" + agent.get_file_name() + "_result.pt"
    Path(file_name).touch()
    file = open(file_name, "w")
    file.write("Highest win rate last 500: " + str(highest_win_rate) + "\n")
    file.write("End win rate: " +  str(win_rate) + "\n")
    file.write("Wins: " + str(p1wins) + "\n")
    file.write("Loses: " + str(p2wins) + "\n")
    file.write("Games played: " + str(games_played) + "\n")
    file.close()
