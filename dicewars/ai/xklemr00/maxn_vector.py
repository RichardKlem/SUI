import sys
import datetime
import statistics

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.xklemr00.maxn_alg import MaxN

# NN stuff
import dicewars.ai.xklemr00.dggraphnet
MODEL_PATHNAME = "dicewars/ai/xklemr00/model.pth"

TRAINING = True

class AI:

    # player_name     the name of the player this AI will control
    # board           an instance of dicewars.client.game.Board
    # players_order   in what order do players take turns
    # max_transfers   number of transfers allowed in a single turn
    def __init__(self, player_name, board, players_order, max_transfers):
        # initialize model
        self.nn_model = dicewars.ai.xklemr00.dggraphnet.DGGraphNet(
            4*34*34+12,
            34,
            12,
            12,
            6)
        # self.nn_model.load(MODEL_PATHNAME)

        self.player_name = player_name
        self.players_order = players_order
        self.maxn = MaxN(player_name, players_order, self.nn_model)
        self.tranfers_todo = []
        self.max_transfers = max_transfers
        self.fisher_increment = 0.25
        self.time_per_leaf_node = 1 # in seconds
        self.first_run = True


        # make records for training
        self.turn_counter = 0
        self.records = []


    def find_best_transfer_recursive(self, board, current_area, transfers_left, found_dices, needed_dices, already_in_path):
        max_found_dices = -1
        max_transfers_left = -1
        best_path = []

        # find all neighbours which are NOT borders
        for neighbour_name in current_area.get_adjacent_areas_names():
            neighbour = board.get_area(neighbour_name)

            # we dont want to move dices from border areas or already visited areas
            if not board.is_at_border(neighbour) and neighbour_name not in already_in_path:

                # stop searching if we found enough dices or run out of transfers
                if (found_dices + neighbour.get_dice() - 1) >= needed_dices or transfers_left == 1:
                    # returns run_out_of_tranfers, found_dices, [(source, target)]
                    return transfers_left - 1, found_dices + neighbour.get_dice() - 1, [(neighbour.get_name(), current_area.get_name())]

                path_transfers_left, path_found_dices, path  = self.find_best_transfer_recursive(board, neighbour, transfers_left-1, found_dices + neighbour.get_dice() - 1, needed_dices, already_in_path + [neighbour_name])

                # check if this path was the best yet
                if (path_transfers_left > max_transfers_left or
                    (path_transfers_left == max_transfers_left and path_found_dices > max_found_dices)):
                    best_path = path + [(neighbour.get_name(), current_area.get_name())]
                    max_found_dices = path_found_dices
                    max_transfers_left = path_transfers_left

        return (max_transfers_left, max_found_dices, best_path)


    # board             an instance of dicewars.client.game.Board
    # transfers_left    number of transfers left in this turn
    # return a list of tuples (source, target) of tranfers to be done
    #   item on index 0 is the first to be done
    def move_dices_to_border(self, board, transfers_left):

        weakest_border_area = None
        weakest_border_value = sys.maxsize

        for border_area in board.get_player_border(self.player_name):
            neighbours_dices = []
            border_area_have_only_border_neighbours = True

            # find all neighbours which are borders
            for neighbour_name in border_area.get_adjacent_areas_names():
                neighbour = board.get_area(neighbour_name)

                if board.is_at_border(neighbour):
                    neighbours_dices += [neighbour.get_dice()]
                else:
                    border_area_have_only_border_neighbours = False

            # we dont have any area to tranfer from
            if border_area_have_only_border_neighbours:
                continue

            # calculate mean of surround areas and current area (2x weight)
            current_border_value = statistics.mean(neighbours_dices + [border_area.get_dice()]*2)

            # overwrite the weakest border if the current one is the weakest
            if current_border_value < weakest_border_value:
                weakest_border_area = border_area
                weakest_border_value = current_border_value

        if weakest_border_area == None:
            return []

        # limit transfers to 3 per weakest border area
        max_transfers_left, max_found_dices, best_path = self.find_best_transfer_recursive(board, weakest_border_area, min(transfers_left, 3), 0, 8 - weakest_border_area.get_dice(), [weakest_border_area.get_name()])

        return best_path


    # board                   an instance of dicewars.client.game.Board
    # nb_moves_this_turn      number of attacks made in this turn
    # nb_transfers_this_turn  number of transfers made in this turn
    # nb_turns_this_game      number of turns ended so far
    # previous_time_left      time (in seconds) left after last decision making
    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        time_start = datetime.datetime.now()

        # for each transfer attack once
        if nb_moves_this_turn >= self.max_transfers:
            return EndTurnCommand()

        # use every transfer, before starting a maxn/monte carlo algorithm if possible
        if not self.tranfers_todo and nb_transfers_this_turn < self.max_transfers:
            self.tranfers_todo = self.move_dices_to_border(board, self.max_transfers - nb_transfers_this_turn)

        if self.tranfers_todo:
            source_name, target_name = self.tranfers_todo.pop(0)
            return TransferCommand(source_name, target_name)

        turn_type, source, target = self.maxn.calculate_best_turn(board)

        time_delta = datetime.datetime.now() - time_start
        time_delta_s = time_delta.microseconds / 1000000

        # calculate how many leaf nodes should be inspected in future run
        if self.first_run:
            self.time_per_leaf_node = time_delta_s / self.maxn.inspected_leaf_nodes
            self.first_run = False

        time_left -= time_delta_s
        monte_carlo_max_leaf_nodes = (time_left*0.9 / self.time_per_leaf_node) / self.max_transfers
        self.maxn.monte_carlo_max_leaf_nodes = min(monte_carlo_max_leaf_nodes, 5000) # for depth == 4 is 5k more than enough

        if turn_type == "attack":
            return BattleCommand(source.get_name(), target.get_name())
        elif turn_type == "transfer":
            return TransferCommand(source.get_name(), target.get_name())
        else:  # turn_type == "end"

            # propagate NN inputs and outputs
            if TRAINING:

                self.turn_counter += 1

                for item in self.maxn.records:
                    self.records.append(
                        [
                            item[0],                # NN input
                            item[1],                # NN output
                            self.turn_counter,      # turn counter for referecnce
                            board.get_player_dice(self.player_name),
                                                    # dice metric to evaluate                           
                        ]
                    )

                self.maxn.records = []

            self.maxn.cooldown = False
            return EndTurnCommand()
