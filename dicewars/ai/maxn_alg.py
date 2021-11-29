import copy
import sys
import math
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.client.game.board import Board
import numpy as np
from copy import deepcopy
import random
import statistics

# MUST be optimized to achieve the best performance!
SCORE_WEIGHT = 5  # size of the biggest region
REGIONS_WEIGHT = 10  # number of regions
AREAS_WEIGHT = 15  # number of areas (region contains areas)
BORDER_FILLING_WEIGHT = 3  # number which indicates how full are border areas
BORDERS_WEIGHT = 3  # number of border areas
NEIGHBOURS_WEIGHT = 1  # number which indicates how well connected are regions

MAXN_MAX_DEPTH = 2
MONTE_CARLO_MAX_DEPTH = 4

class MaxN:

    # time_left?
    def __init__(self, player_name, players_order):
        self.player_index = players_order.index(player_name)
        self.players_order = players_order
        self.player_name = player_name
        self.monte_carlo_max_leaf_nodes = 200
        self.all_inspected_nodes = 0
        self.all_inspected_leaf_nodes = 0

    def deep_copy_board(self, board):
        # just to correctly create a Board object
        board_copy = copy.deepcopy(board)

        # creates a deep copy of each Area, because the previous deepcopy couldnt reach it
        for key, area in board.areas.items():
            board_copy.areas[key] = copy.deepcopy(area)

        return board_copy

    def forward_pruning_possible_attacks(self, board, player_name):
        attacks_to_inspect = []

        # forward pruning at depth 1 (our next turn)
        for source, target in possible_attacks(board, player_name):
            probability_of_success = probability_of_successful_attack(board, source.get_name(), target.get_name())
            probability_of_holding = probability_of_holding_area(board, target.get_name(), target.get_dice(), player_name)
            both_areas_have_8 = target.get_dice() == 8 and target.get_dice() == 8

            # do not attack if:
            #   probability of success is lower than 65% and we have less than 4 dices
            #   probability of holding the area until next turn is lower than 30 % (2 players), 40 % (4 players), ...
            if ((probability_of_success < 0.55 and source.get_dice() < 4) or
                (probability_of_holding < (0.20 + len(self.players_order)*0.05) and not both_areas_have_8)):
                continue
            else:
                attacks_to_inspect += [(source, target)]

        return attacks_to_inspect

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
        if len(player_areas) == 0:
            return 0

        # but we really want to get to node where we won
        if len(player_areas) == len(board.areas):
            return sys.maxsize

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

        # use only sum of vector instead of length, because DNN can simulate this internally
        return sum(vector)

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
                node_value[index] = self.evaluate_current_node(self.players_order[index], board)
            return node_value

        next_player_index = (player_index + 1) % len(self.players_order)

        # player can end his turn
        # since we dont take into account a randomised addition of dice -- the board wont change
        endturn_value = self.maxn_recursive(board, depth + 1, next_player_index)

        # the more areas AI have (compared to all areas) the more it tries to attack, instead of ending the turn
        # to force AI to attack we penalize end turn value max -10%
        if self.player_name == self.players_order[player_index]:
            areas_owned_ratio = len(board.get_player_areas(self.player_name)) / len(board.areas)

            # start at ratio > 50%, otherwise do not penalize end turn
            if areas_owned_ratio > 0.5:
                for index, name in enumerate(self.players_order):
                    endturn_value[index] *= 1 - (areas_owned_ratio*0.1)

        current_level_evaluation = [endturn_value]
        moves = [("end", None, None)]

        for source, target in possible_attacks(board, self.players_order[player_index]):
            # if we do not succed with the attack it wont help us in any way so we will just ignore this attack
            probability_of_success = probability_of_successful_attack(board, source.get_name(), target.get_name())
            probability_of_holding = probability_of_holding_area(board, target.get_name(), target.get_dice(), self.players_order[player_index])
            both_areas_have_8 = target.get_dice() == 8 and target.get_dice() == 8

            # do not attack if:
            #   probability of success is lower than 65% and we have less than 4 dices
            #   probability of holding the area until next turn is lower than 30 % (2 players), 40 % (4 players), ...
            if ((probability_of_success < 0.55 and source.get_dice() < 4) or
                (probability_of_holding < (0.20 + len(self.players_order)*0.05) and not both_areas_have_8)):
                continue

            # makes a deep copy of a board
            # OPTIMIZE: it is not necessary to create a copy since it is not running in parallel,
            # just make a move and then reverse it
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

            # calculate an evalutation of a current node
            node_value = [a + b for a, b in zip(success_value, unsuccess_value)]
            current_level_evaluation += [node_value]
            moves += [("attack", source, target)]

        max_value = current_level_evaluation[0]
        next_move = moves[0]

        for index, value in enumerate(current_level_evaluation):
            if value[player_index] > max_value[player_index]:
                max_value = value
                next_move = moves[index]

        if depth == 0:
            return next_move
        else:
            return max_value

    def simulate_path_to_leaf(self, first_attack_source, first_attack_target, board):
        end_turn = [(-1, -1)]

        # make the first move (our move)
        if first_attack_source != -1:
            probability_of_success = probability_of_successful_attack(board, first_attack_source.get_name(), first_attack_target.get_name())
            self.make_attack(board, first_attack_source, first_attack_target, random.random() < probability_of_success)

        player_index = (self.player_index + 1) % len(self.players_order)

        # start from 1 because we already did the first move
        for depth in range(1, MONTE_CARLO_MAX_DEPTH):
            possible_attacks = self.forward_pruning_possible_attacks(board, player_index)

            # we cant attack anymore or we can only do prunned attacks
            if not possible_attacks:
                self.inspected_leaf_nodes += 1
                return self.evaluate_current_node(self.player_name, board)

            source, target = random.choice(possible_attacks + end_turn)

            # if we pick end turn (source == -1) skip update of board
            if source != -1:
                probability_of_success = probability_of_successful_attack(board, source.get_name(), target.get_name())
                # make randomly picked attack with its probability of success
                self.make_attack(board, source, target, random.random() < probability_of_success)

            player_index = (player_index + 1) % len(self.players_order)

        self.inspected_leaf_nodes += 1
        return self.evaluate_current_node(self.player_name, board)

    def monte_carlo(self, board, leaves_to_inspect):
        end_turn = [(-1, -1)]
        moves_to_inspect = self.forward_pruning_possible_attacks(board, self.player_name) + end_turn

        # calculate number of leaves to visit for each attack
        leaves_for_each_move = round(leaves_to_inspect / len(moves_to_inspect))

        moves_evaluation = [0]*len(moves_to_inspect)
        move_index = 0

        # run monte carlo
        for source, target in moves_to_inspect:
            # run "leaves_for_each_move" simulations and return average evaluation
            leaves_evaluations = []

            for leaf in range(0, leaves_for_each_move):
                board_copy = self.deep_copy_board(board)
                leaves_evaluations += [self.simulate_path_to_leaf(source, target, board_copy)]
            moves_evaluation[move_index] = statistics.mean(leaves_evaluations)
            move_index += 1

        # again, penalize end turn if AI have more than half of the map
        areas_owned_ratio = len(board.get_player_areas(self.player_name)) / len(board.areas)
        if areas_owned_ratio > 0.5:
            moves_evaluation[-1] *= 1 - (areas_owned_ratio*0.1)

        max_value = max(moves_evaluation)
        best_move_index = moves_evaluation.index(max_value)
        best_move_src, best_move_target = moves_to_inspect[best_move_index]

        if best_move_src == -1:
            return ("end", best_move_src, best_move_target)
        else:
            return ("attack", best_move_src, best_move_target)

    def calculate_best_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn):
        self.inspected_nodes = 0
        self.inspected_leaf_nodes = 0

        next_move = self.monte_carlo(board, self.monte_carlo_max_leaf_nodes)
        #next_move = self.maxn_recursive(board, 0, self.player_index)

        self.all_inspected_nodes += self.inspected_nodes
        self.all_inspected_leaf_nodes += self.inspected_leaf_nodes

        return next_move
