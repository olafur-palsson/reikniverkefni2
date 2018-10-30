import re

import numpy as np

from agents.agent_interface import AgentInterface
from backgammon_game import Backgammon


def parse_input(s):
    s = s.strip()
    if len(s) == 0:
        return []
    else:
        try:
            input_pair = re.compile("\\s+").split(s)
            pos_from = int(input_pair[0])
            pos_to = int(input_pair[1])
            move = np.array([pos_from, pos_to], dtype=np.int64)
            return move
        except:
            return None




def is_first_move_legal(all_legal_moves, first_move):

    print(all_legal_moves)
    print(first_move)

    if len(all_legal_moves) == 0:
        # No moves to be made
        if len(first_move) == 0:
            return True
    else:
        # Some moves are possible
        for moves in all_legal_moves:
            # Moves [[1, 2], [3, 4]]
            if len(moves) == 1 or len(moves) == 2:
                # 1 legal moves
                move_1 = moves[0]
                if first_move[0] == move_1[0] and first_move[1] == move_1[1]:
                    return True
            elif len(moves) == 0:
                # 0 legal moves
                if len(first_move) == 0:
                    return True
            else:
                raise Exception("This shouldn't have happened!")
    
    return False

def is_second_move_legal(all_legal_moves, first_move, second_move):

    for moves in all_legal_moves:
        # Moves [[1, 2], [3, 4]]
        if len(moves) == 1 or len(moves) == 2:
            # 1 legal moves
            move_1 = moves[0]
            if first_move[0] == move_1[0] and first_move[1] == move_1[1]:
                return True
        elif len(moves) == 0:
            # 0 legal moves
            if len(first_move) == 0:
                return True
        else:
            raise Exception("This shouldn't have happened!")

    return False





class HumanAgent(AgentInterface):

    def __init__(self):
        pass

    def action(self, board, dice, player):
        """
        Args:
            board (ndarray): backgammon board
            dice (ndarray): a pair of dice
            player: the number for the player on the board who's turn it is.

        Returns:
            A move `move`.
        """

        all_legal_moves = Backgammon.get_all_legal_moves_for_two_dice(board, dice)[0]

        # Runs this until a legal move is made.
        while True:

            print(Backgammon.to_string(board))
            print("")
            print("   You: " + str(Backgammon.get_player_symbol(player)))
            print("   Dice: " + str(dice))
            print("")
            print("    Press (enter) to pass if no moves are possible.")
            print("    Syntax: POSITION_FROM POSITION_TO")

            if len(all_legal_moves) == 0:
                # No possible moves
                print("No possible moves.  Press (enter) to continue.")
                input("Input: ")
                return []
            else:
                # Some moves possible
                move_1 = parse_input(input("Input: "))

                valid_move_1 = False

                future_legal_moves = []

                for moves in all_legal_moves:
                    if len(moves) > 0:
                        first_move = moves[0]
                        if len(first_move) == 2:
                            if first_move[0] == move_1[0] and first_move[1] == move_1[1]:
                                valid_move_1 = True
                                future_legal_moves += [moves[1]]
                
                if valid_move_1:
                    # Check if future moves are possible
                    if len(future_legal_moves) == 0:
                        return [move_1]
                    else:
                        move_2 = parse_input(input("Input: "))

                        for second_move in future_legal_moves:
                            if second_move[0] == move_2[0] and second_move[1] == move_2[1]:
                                return [move_1, move_2]
                        
                        print("Invalid second move")
                else:
                    print("Invalid move")




    



