import copy
import sys
import math
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack
from dicewars.client.game.board import Board
import numpy as np
from copy import deepcopy

# MUST be optimized to achieve the best performance!
SCORE_WEIGHT = 5  # size of the biggest region
REGIONS_WEIGHT = 10  # number of regions
AREAS_WEIGHT = 15  # number of areas (region contains areas)
BORDER_FILLING_WEIGHT = 3  # number which indicates how full are border areas
BORDERS_WEIGHT = 3  # number of border areas
NEIGHBOURS_WEIGHT = 1  # number which indicates how well connected are regions

MAXN_MAX_DEPTH = 1

END_PENALTY = 0.55


class MaxN:

    # time_left?
    def __init__(self, player_name, players_order):
        self.player_index = players_order.index(player_name)
        self.players_order = players_order
        self.player_name = player_name
        self.all_inspected_nodes = 0
        self.all_inspected_leaf_nodes = 0

    def deep_copy_board(self, board):
        # just to correctly create a Board object
        board_copy = copy.deepcopy(board)

        # creates a deep copy od each Area, because the previous deepcopy couldnt reach it
        for key, area in board.areas.items():
            board_copy.areas[key] = copy.deepcopy(area)

        return board_copy

    # returns a float number which indicates how good is this node for player "player_name"
    def evaluate_current_node(self, player_name, board):
        all_regions = board.get_players_regions(player_name)
        player_areas = board.get_player_areas(player_name)

        border_filling_sum = 0
        border_areas_num = 0
        for border_area in board.get_player_border(player_name):
            border_filling_sum += border_area.get_dice()
            border_areas_num += 1

        possible_transfers_num = 0
        for area in player_areas:
            for neighbour in area.get_adjacent_areas_names():
                neighbour = board.get_area(neighbour)
                if neighbour.get_owner_name() == player_name:
                    possible_transfers_num += 1

        # we really dont want to get to node where we lost
        if len(player_areas) == 0 or border_areas_num == 0:
            return -1

        score = len(max(all_regions, key=len)) / len(board.areas)
        regions = 1 - (len(all_regions) / len(player_areas))
        areas = len(board.get_player_areas(player_name)) / len(board.areas)
        border_filling = border_filling_sum / (border_areas_num*8)
        borders = border_areas_num
        neighbours = possible_transfers_num

        vector = [score * SCORE_WEIGHT,
                  regions * REGIONS_WEIGHT,
                  areas * AREAS_WEIGHT,
                  border_filling * BORDER_FILLING_WEIGHT,
                  borders * BORDERS_WEIGHT,
                  neighbours * NEIGHBOURS_WEIGHT]

        return math.sqrt(np.dot(vector, vector))

    def make_attack(self, board, source, target, attack_success):
        if attack_success:
            # overwrite dices in target area
            board.areas[str(target.get_name())].set_dice(source.get_dice() - 1)
            # set dices of source to 1
            board.areas[str(source.get_name())].set_dice(1)
            # overwrite owner of target area
            board.areas[str(target.get_name())].owner_name = source.get_owner_name()
        else:
            # if attacker have 8 dices -- defender loses 2 dices
            #                  4-7 dices -- defender loses 1 dice
            remove_defenders_dices = source.get_dice() // 4
            board.areas[str(target.get_name())].set_dice(max(target.get_dice() - remove_defenders_dices, 1))
            board.areas[str(source.get_name())].set_dice(1)


    def maxn_recursive(self, board, depth, player_index):
        self.inspected_nodes += 1
        # we have reached a leaf node, just return its value
        if depth >= MAXN_MAX_DEPTH:
            self.inspected_leaf_nodes += 1
            # evaluate the node for all players and return an array of these values
            node_value = [0] * len(self.players_order)

            for index, name in enumerate(self.players_order):
                node_value[index] = self.evaluate_current_node(self.players_order[player_index], board)
            return node_value

        next_player_index = (player_index + 1) % len(self.players_order)

        # player can end his turn
        # since we dont take into account a randomised addition of dice -- the board wont change
        endturn_value = self.maxn_recursive(board, depth + 1, next_player_index)
        endturn_value = [x * END_PENALTY for x in endturn_value]
        deeper_level_evaluation = [endturn_value]
        moves = [("end", None, None)]

        for source, target in possible_attacks(board, self.players_order[player_index]):
            # if we do not succed with the attack it wont help us in any way so we will just ignore this attack
            probability_of_success = probability_of_successful_attack(board, source.get_name(), target.get_name())
            if probability_of_success < 0.55 and source.get_dice() < 4:
                continue
            # makes a deep copy of a board
            board_copy = self.deep_copy_board(board)
            self.make_attack(board_copy, source, target, True)
            success_value = self.maxn_recursive(board_copy, depth + 1, next_player_index)
            for index, name in enumerate(self.players_order):
                success_value[index] *= probability_of_success

            board_copy = self.deep_copy_board(board)
            self.make_attack(board_copy, source, target, False)
            unsuccess_value = self.maxn_recursive(board_copy, depth + 1, next_player_index)
            for index, name in enumerate(self.players_order):
                unsuccess_value[index] *= 1 - probability_of_success

            # needs more experimenting to improve an evaluation of an attack
            # e.g. min, avg, ...
            attack_value = max(success_value, unsuccess_value)
            deeper_level_evaluation += [attack_value]
            moves += [("attack", source, target)]

        node_value = []
        max_value = -1
        next_move = None

        #print(deeper_level_evaluation, file=sys.stderr)
        #print("", file=sys.stderr)

        for index, value in enumerate(deeper_level_evaluation):
            if value[player_index] > max_value:
                max_value = value[player_index]
                node_value = value
                next_move = moves[index]

        if depth == 0:
            return next_move
        else:
            return node_value

    def calculate_best_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn):
        self.inspected_nodes = 0
        self.inspected_leaf_nodes = 0
        next_move = self.maxn_recursive(board, 0, self.player_index)

        self.all_inspected_nodes += self.inspected_nodes
        self.all_inspected_leaf_nodes += self.inspected_leaf_nodes

        return next_move
