from dicewars.client.game.board import Board
from dicewars.client.game.area import Area

from dicewars.ai.utils import probability_of_successful_attack

import torch
import math

# implementation of the graph convolution layers has been lifted from
# the PyCGN implementation, available at https://github.com/tkipf/pygcn 

class DGGraphConv(torch.nn.Module):
    
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.in_size  = in_size
        self.out_size = out_size

        self.weight = torch.nn.Parameter(torch.Tensor(in_size, out_size))
        self.bias   = torch.nn.Parameter(torch.Tensor(1, out_size))

        self.reset_parameters()

    # lifted from PyCGN
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    # lifted from PyCGN
    def forward(self, input, adjacency):
        supp = torch.mm(input, self.weight)
        out  = torch.spmm(adjacency, supp)
        return out + self.bias


class DGGraphNet(torch.nn.Module):

    def __init__(self, in_size, mat_size, par_size, hid_size, out_size
       ) -> None:
        # in_size:          total size of input vector (2*34*34+12)
        # mat_size:         side of board matrix (34)
        # par_size:         size of parameters vector (6)
        # hid_size:         size of hidden layer vector (varies)
        # out_size:         output size (6)

        super().__init__()

        self.in_size  = in_size
        self.mat_size = mat_size
        self.par_size = par_size
        self.out_size = out_size


        # board -> GCN -> FC -+
        #                     |
        #                     V
        # features -> FC -> concat -> FC -> out

        # two channels for GCN
        self.gcn = DGGraphConv(mat_size, hid_size)


        # fully connected post GCN
        self.fc1 = torch.nn.Linear(mat_size*hid_size, hid_size)

        # fully connected features
        self.fc2 = torch.nn.Linear(par_size, par_size)

        # fully connected output
        self.fc3 = torch.nn.Linear(hid_size+par_size, out_size)

        # softmax out
        self.sm  = torch.nn.LogSoftmax(dim=0)



    def forward(self, input):
        
        # split input into board and global features
        # matrices
        board = input[:-self.par_size].reshape(2, self.mat_size, self.mat_size)
        own = board[0]
        opp = board[1]

        # features
        feat = input[-self.par_size:]

        # push board through
        # GCN
        conv = torch.nn.functional.relu(self.gcn(own, opp)).reshape(-1)

        # FC
        conv_fc = torch.nn.functional.relu(self.fc1(conv))

        # push features through
        # FC
        feat_fc = torch.nn.functional.relu(self.fc2(feat))

        # concatenate outputs
        conv_feat = torch.cat([conv_fc, feat_fc])

        # push through FC
        out = self.fc3(conv_feat)

        # return output
        return torch.exp(self.sm(out))


    # save model parameters into file between instances
    def save(self, pathname):
        torch.save(self.state_dict(), pathname)

    # load parameters into model afer initiaization
    def load(self, pathname):
        self.load_state_dict(torch.load(pathname))


def build_nn_input(board: Board, player_name: int):

    boardOut = torch.zeros((2,34,34))
    

    # input format
    # 34x34 matrices + global parameters
    # matrices:
    # - 1 with player strength across the diagonal and connected areas?
    # - 3 with neighboring areas for each player and their respective strength
    #
    # global parameters:
    # - area count, largest area, army strength -> per player

    # all areas are relevant:
    for player in range(1,3):
        for pArea in board.get_player_areas(player):
            n = pArea.name

            if player == player_name:
                # 1) put area strength into ch1 diagonal
                # 2) add own adjacencies into ch1
                # 3) add adjacencies with opponent into ch2
                boardOut[0][n][n] = 1 + pArea.dice
                for surrAreaN in pArea.get_adjacent_areas_names():
                    if (surrAreaN > n): continue
                    # if adjacent area also belongs to player, add it to channel 1
                    surrArea = board.get_area(surrAreaN)
                    adjOwner = surrArea.get_owner_name()
                    if (adjOwner == player_name):
                        boardOut[0][n][surrAreaN] = 1
                    else:
                        boardOut[1][n][surrAreaN] = probability_of_successful_attack(board, n, surrAreaN)
            else:
                # 1) put strength onto diagonal
                # 2) add adjacencies with opponent into ch2
                boardOut[1][n][n] = 1 + pArea.dice
                for surrAreaN in pArea.get_adjacent_areas_names():
                    if (surrAreaN > n): continue
                    surrArea = board.get_area(surrAreaN)
                    adjOwner = surrArea.get_owner_name()
                    if (adjOwner == player_name):
                        boardOut[1][surrAreaN][n] = probability_of_successful_attack(board, surrAreaN, n)

    # global parameters:
    paramOut = torch.zeros(6)
    
    j = 3
    for player in range(1, 3):
        # gets player's areas count
        areaCount = len(board.get_player_areas(player))
        # gets player's dice count
        armyStrength = board.get_player_dice(player)
        
        # gets player largest region
        playerRegions = board.get_players_regions(player)
        largestArea = 0
        for region in playerRegions:
            if len(region) > largestArea:
                largestArea = len(region)
        
        # assigns the values of our AI to the first three positions of the matrix
        if player == player_name:
            paramOut[0] = areaCount
            paramOut[1] = armyStrength
            paramOut[2] = largestArea
        else:
            paramOut[j] = areaCount
            j = j + 1
            paramOut[j] = armyStrength
            j = j + 1
            paramOut[j] = largestArea
            j = j + 1
    
    return torch.cat([boardOut.reshape(-1), paramOut])

    



        
if __name__=="__main__":
    exit(0)