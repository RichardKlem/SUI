import logging
from ..utils import possible_attacks

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand


class AI:
    """Agent using Strength Difference Checking (SDC) strategy

    This agent prefers moves with highest strength difference
    and doesn't make moves against areas with higher strength.
    """
    def __init__(self, player_name, board, players_order, max_transfers):
        """
        Parameters
        ----------
        game : Game

        Attributes
        ----------
            Areas that can make an attack
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.max_transfers = max_transfers

        self.stage = 'attack'

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Creates a list with all possible moves along with associated strength
        difference. The list is then sorted in descending order with respect to
        the SD. A move with the highest SD is then made unless the highest
        SD is lower than zero - in this case, the agent ends its turn.
        """

        if self.stage == 'attack':
            attacks = []
            for source, target in possible_attacks(board, self.player_name):
                area_dice = source.get_dice()
                strength_difference = area_dice - target.get_dice()
                attack = [source.get_name(), target.get_name(), strength_difference]
                attacks.append(attack)

            attacks = sorted(attacks, key=lambda attack: attack[2], reverse=True)

            if attacks and attacks[0][2] >= 0:
                return BattleCommand(attacks[0][0], attacks[0][1])
            else:
                self.stage = 'transfer'

        if self.stage == 'transfer':
            if nb_transfers_this_turn < self.max_transfers:
                border_names = [a.name for a in board.get_player_border(self.player_name)]
                all_areas = board.get_player_areas(self.player_name)
                inner = [a for a in all_areas if a.name not in border_names]

                for area in inner:
                    if area.get_dice() < 2:
                        continue

                    for neigh in area.get_adjacent_areas_names():
                        if neigh in border_names and board.get_area(neigh).get_dice() < 8:
                            return TransferCommand(area.get_name(), neigh)
            else:
                self.logger.debug(f'Already did {nb_transfers_this_turn}/{self.max_transfers} transfers, skipping further')

        self.stage = 'attack'
        return EndTurnCommand()
