import random
import logging
import sys

from dicewars.ai.utils import possible_attacks
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.maxn_alg import MaxN


class AI:
    """TODO
    """

    # player_name     the name of the player this AI will control
    # board           an instance of dicewars.client.game.Board
    # players_order   in what order do players take turns
    # max_transfers   number of transfers allowed in a single turn
    def __init__(self, player_name, board, players_order, max_transfers):
        """
        Parameters
        ----------
        game : Game
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.log_file = open("debug.save", "w")
        self.players_order = players_order
        self.maxn = MaxN(player_name, players_order)


    # board                   an instance of dicewars.client.game.Board
    # nb_moves_this_turn      number of attacks made in this turn
    # nb_transfers_this_turn  number of transfers made in this turn
    # nb_turns_this_game      number of turns ended so far
    # previous_time_left      time (in seconds) left after last decision making
    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        print("Turn: %s" % nb_turns_this_game, file=sys.stderr)

        # use every transfer, before starting a maxn algorithm if possible

        turn_type, source, target = self.maxn.calculate_best_turn(board, nb_moves_this_turn, nb_transfers_this_turn)

        print("Action: %s" % turn_type, file=sys.stderr)
        if turn_type == "attack":
            return BattleCommand(source.get_name(), target.get_name())
        elif turn_type == "transfer":
            return TransferCommand(source.get_name(), target.get_name())
        else:  # turn_type == "end"
            return EndTurnCommand()
