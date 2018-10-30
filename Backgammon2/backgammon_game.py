#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class implementation of backgammon.

NOTE:
Player 1 is defined as the player who has the first turn.
Think of player 1 as the current player.

If add a mechanism in the beginning of the game such that each player rolls
to determine who's to the play, then it might be necessar to flip the board. 


TODO: it would be cool if it were possible to hot-swap players, i.e. in the
middle of a game swap one player (e.g. some neural network player) for a 
human player.

TODO: add ability to support human players.  So both players can be occupied
by huamsn.

TODO: add the ability to record an episode.  We could possibly generate 
episodes generated by humans, and use off-policy training to train other
agents.
"""
import numpy as np
from pathlib import Path


def str_symbol(board, pos, height):
    """
    A helper class for the string representation for `Backgammon`.

    Args:
        board (ndarray): backgammon board
        pos (int): ?
        height: (int) ?

    Returns:
        Symbols "x", "o" or " ".
    """

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

    PLAYER_1_SYMBOL = "x"
    PLAYER_2_SYMBOL = "o"

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

        Args:
            player: ?
        """
        self.player_1 = player


    def set_player_2(self, player):
        """
        Sets player 2 to `player`.

        Args:
            player: ?
        """
        self.player_2 = player


    def reset(self):
        """
        Initializes the board and resets everything to default.
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
        

        # board[0]  unknown

        self.board = np.zeros(29)
        self.board[1] = -2
        self.board[6] = 5
        self.board[8] = 3
        self.board[12] = -5
        self.board[13] = 5
        self.board[17] = -3
        self.board[19] = -5
        self.board[24] = 2

        # board[25]  jail for player 1
        # board[26]  jail for player 2
        # board[27]  off-ed checkers for player 1
        # board[28]  off-ed checkers for player 2

        # Reset active player
        # `1` for player 1 and `-1` for player 2
        self.active_player = 0  # Just to get type information


    def roll_dice(self):
        """
        Rolls the dice, i.e. creates two numbers, each number is sampled
        from the set {1, 2, 3, 4, 5, 6}.

        Returns:
            A `numpy.ndarray` of two numbers.
        """
        return np.random.randint(1,7,2)


    def is_game_over(self):
        """
        Checks whether the game is over or not.

        Returns:
            `True` if the game is over, otherwise `False`.
        """
        return self.board[27] == 15 or self.board[28] == -15


    def get_winner(self):
        """
        Returns the number of the player that won.

        Returns:
            `1` if player 1 won, `-1` if player 2 won, otherwise `0`.
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

        Args:
            board: the board
        """
        
        if board is None:
            board = self.board

        string = str(np.array2string(board[1:13])+'\n'+
                     np.array2string(board[24:12:-1])+'\n'+
                     np.array2string(board[25:29]))
        print(string)


    def get_flipped_board(self):
        """
        Creates a flipped board without affecting the board of this game.

        Returns:
            A flipped board (ndarray).
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
        "Flips" the board in-place.
        """

        self.board = self.get_flipped_board()
    

    def clone(self):
        """
        Creates a clone of this backgammon game.
        """
        pass


    def copy_board(self):
        """
        Create a copy of this backgammon board.

        Returns:
            A copy of the board (ndarray) of this game in its current state.
        """
        return np.copy(self.board)


    def play(self, commentary = False, verbose = False):
        """
        Make player 1 and player 2 play a game of backgammon.

        When a player picks an action (s)he has 

        Args:
            commentary (bool): Whether to include commentary `True`, or not `False`.

        Returns:
            `1` if player 1 won or `-1` if player 2 won.
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

            # If you roll of your dice is, say 5-5, then you use 5 four times,
            # or the pair twice.
            turns = 1
            if dice[0] == dice[1]:
                turns = 2

            # np.array([1], dtype=np.int64)

            # TODO: make sure the same die is not reused.

            if verbose:
                print("====================================================================")


            # Make a move (2 moves if the same number appears on the dice)
            for i in range(turns):

                if verbose:
                    print("Dice:", dice)
                    print("")
                    print("BEFORE")
                    print(self)

                board_copy = self.copy_board()

                move = None

                if self.active_player == 1:
                    move = self.player_1.action(board_copy, dice, 1)
                elif self.active_player == -1:
                    move = self.player_2.action(board_copy, dice, 1)
                else:
                    raise Exception("This shouldn't have happened.")

                # TODO figure out which die was used

                # update the board
                if len(move) != 0:
                    for m in move:
                        self.update_board(m)

                if commentary:
                    print("move from player", self.active_player, ":")
                    #print("board:")
                    #if self.active_player == 1:
                    #    self.pretty_print()
                    #else:
                    #    self.pretty_print(self.get_flipped_board())

                if verbose:
                    print("AFTER")
                    print(self)

            # players take turns
            self.active_player = -self.active_player
            self.flip_board()

        return -self.active_player


    def update_board(self, move):
        """
        Updates the board by making the move `move`.  A `moves` moves one
        checker from some position to another.  If the checker lands on 
        a blot (point occupied by one opposing checker) then the opposing
        checker is moved to the "jail".

        Args:
            move (ndarray): a pair of numbers, the first one is *from* and second is *to*.
        """

        # If we can make a move, we move.
        if len(move) > 0:

            startPip = move[0]  # from
            endPip = move[1]    # to

            # moving the dead piece if the move kills a piece
            if self.board[endPip] == -1:
                self.board[endPip] = 0
                jail = 26
                self.board[jail] = self.board[jail] - 1

            self.board[startPip] = self.board[startPip]-1
            self.board[endPip] = self.board[endPip]+1


    def __str__(self):
        """
        If you evaluate `str(x)` where `x` is an instance of `Backgammon` then
        this string (which is constructed here) is returned.

        Returns:
            A `str` representing the board of this game, and all relevant
            information.
        """
        return Backgammon.to_string(self.board)


    @staticmethod
    def to_string(board):
        """
        Returns a CLI representation of the game

        Args:
            board (ndarray): the board

        Returns:
            A CLI representation (string) of the game.
        """

        msg = ""
        height = int(np.amax([abs(x) for x in board]))

        dir_banner = "        <-- x --                                       -- o -->"

        top_banner = "   .---.-------------.-------------.-------------.-------------.---.---.---.---."

        bottom_banner = "   '-?-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-^-^-^-^-^-^-'-J-'-J-'-O-'-O-'"
        bottom_banner_2 = "                                                                 1   2   1   2"

        num_banner_1 = "                             1 1 1   1 1 1 1 1 1   1 2 2 2 2 2   2   2   2   2"
        num_banner_2 = "     0   1 2 3 4 5 6   7 8 9 0 1 2   3 4 5 6 7 8   9 0 1 2 3 4   5   6   7   8"

        msg += dir_banner
        msg += "\n"
        msg += top_banner
        msg += "\n"
        for i in range(height):
            submsg = ""
            j = height - i
            h = j - 1

            if j < 10:
                submsg += " " + str(j)
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
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 7, h)
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
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 13, h)
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
            submsg += "|"
            submsg += " "
            submsg += str_symbol(board, 19, h)
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
            submsg += "|"
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
            submsg += " "
            submsg += str_symbol(board, 28, h)
            submsg += " "
            submsg += "|"
            submsg += " "

            submsg += "\n"

            msg += submsg
    
        msg += bottom_banner
        msg += "\n"
        msg += bottom_banner_2
        msg += "\n"
        msg += "\n"
        msg += num_banner_1
        msg += "\n"
        msg += num_banner_2
        msg += "\n"

        return msg


    @staticmethod
    def check_if_game_is_over(board):
        """
        Checks whether the game for board `board` is over or not.

        Args:
            board (ndarray): the backgammon board

        Returns:
            `True` if the game is over, otherwise `False`.
        """
        return board[27] == 15 or board[28] == -15


    @staticmethod
    def get_updated_board(board, move):
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

    
    @staticmethod
    def get_player_symbol(player):
        if player == 1:
            return Backgammon.PLAYER_1_SYMBOL
        elif player == -1:
            return Backgammon.PLAYER_2_SYMBOL
        else:
            raise Exception("Shouldn't have happened!")

    
    @staticmethod
    def get_all_legal_moves_for_one_die(board, die):
        """
        Finds all legal moves for the backgammon board `board` and one die.

        Args:
            board (ndarray): the backgammon board
            die: the number of the die (and which player is up?)
    
        Returns:
            All possible moves for this die.
        """
    
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
                elif not Backgammon.check_if_game_is_over(board): # smá fix
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


    @staticmethod
    def get_all_legal_moves_for_two_dice(board, dice):
        """
        Finds all possible moves and their corresponding after-states.

        NOTE: an after-state is simply what the board will look like after making
        the move.

        Args:
            board (ndarray): backgammon board
            dice (ndarray): the dice rolled by player (and which player is up?)

        Returns:
            A tuple of `moves` (if they exists) and their corresponding after-states `boards`.
        """
        # finds all possible moves and the possible board after-states
        # inputs are the BG-board, the dices rolled and which player is up
        # outputs the possible pair of moves (if they exists) and their after-states

        moves = []
        boards = []

        # try using the first dice, then the second dice
        possible_first_moves = Backgammon.get_all_legal_moves_for_one_die(board, dice[0])
        for m1 in possible_first_moves:
            temp_board = Backgammon.get_updated_board(board,m1)
            possible_second_moves = Backgammon.get_all_legal_moves_for_one_die(temp_board,dice[1])
            for m2 in possible_second_moves:
                moves.append(np.array([m1,m2]))
                boards.append(Backgammon.get_updated_board(temp_board,m2))

        if dice[0] != dice[1]:
            # try using the second dice, then the first one
            possible_first_moves = Backgammon.get_all_legal_moves_for_one_die(board, dice[1])
            for m1 in possible_first_moves:
                temp_board = Backgammon.get_updated_board(board,m1)
                possible_second_moves = Backgammon.get_all_legal_moves_for_one_die(temp_board,dice[0])
                for m2 in possible_second_moves:
                    moves.append(np.array([m1,m2]))
                    boards.append(Backgammon.get_updated_board(temp_board,m2))

        # if there's no pair of moves available, allow one move:
        if len(moves)==0:
            # first dice:
            possible_first_moves = Backgammon.get_all_legal_moves_for_one_die(board, dice[0])
            for m in possible_first_moves:
                moves.append(np.array([m]))
                boards.append(Backgammon.get_updated_board(temp_board,m))

            # second dice:
            possible_first_moves = Backgammon.get_all_legal_moves_for_one_die(board, dice[1])
            for m in possible_first_moves:
                moves.append(np.array([m]))
                boards.append(Backgammon.get_updated_board(temp_board,m))

        return moves, boards
        

