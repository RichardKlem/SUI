from dicewars.client.game.board import Board
from dicewars.client.game.area import Area

from dicewars.ai.utils import probability_of_successful_attack

import torch
import math

# implementation of the graph convolution layers has been lifted from
# the PyCGN implementation, available at https://github.com/tkipf/pygcn 

class DGGraphConv(torch.nn.Module):
    ...
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
    ...

    def __init__(self, in_size, mat_size, par_size, hid_size, out_size
       ) -> None:
        # in_size:          total size of input vector (4*34*34+12)
        # mat_size:         side of board matrix (34)
        # par_size:         size of parameters vector (12)
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

        # three channels for GCN
        self.gcn1 = DGGraphConv(mat_size, hid_size)
        self.gcn2 = DGGraphConv(mat_size, hid_size)
        self.gcn3 = DGGraphConv(mat_size, hid_size)

        # fully connected post GCN
        self.fc1 = torch.nn.Linear(3*mat_size*hid_size, hid_size)

        # fully connected features
        self.fc2 = torch.nn.Linear(par_size, par_size)

        # fully connected output
        self.fc3 = torch.nn.Linear(hid_size+par_size, out_size)

        # softmax out
        self.sm  = torch.nn.LogSoftmax(dim=0)



    def forward(self, input):
        
        # split input into board and global features
        # matrices
        board = input[:-self.par_size].reshape(4, self.mat_size, self.mat_size)
        main = board[0]
        adj1 = board[1]
        adj2 = board[2]
        adj3 = board[3]

        # features
        feat = input[-self.par_size:]

        # push board through
        # GCN
        conv1 = torch.nn.functional.relu(self.gcn1(main, adj1))
        conv2 = torch.nn.functional.relu(self.gcn2(main, adj2))
        conv3 = torch.nn.functional.relu(self.gcn3(main, adj3))

        conv = torch.cat([conv1, conv2, conv3]).reshape(-1)


        # FC
        conv_fc = torch.nn.functional.relu(self.fc1(conv))

        # push features through
        # FC
        feat_fc = torch.nn.functional.relu(self.fc2(feat))

        # concatenate outputs
        conv_feat = torch.cat([conv_fc, feat_fc])

        # push through FC
        out = self.sm(self.fc3(conv_feat))

        # return output
        return torch.exp(out)


def build_nn_input(board: Board, player_name: int):

    boardOut = torch.zeros((4,34,34))
    

    # input format
    # 34x34 matrices + global parameters
    # matrices:
    # - 1 with player strength across the diagonal and connected areas?
    # - 3 with neighboring areas for each player and their respective strength
    #
    # global parameters:
    # - area count, largest area, army strength -> per player

    # only player areas are relevant:
    for pArea in board.get_player_areas(player_name):
        n = pArea.name

        # set diagonal to 1 + strength
        x[0][n][n] = 1 + pArea.dice

        # find connected areas
        for surrAreaN in pArea.get_adjacent_areas_names():


            # if adjacent area also belongs to player, add it to channel 1
            surrArea = board.get_area(surrAreaN)
            adjOwner = surrArea.get_owner_name()
            if (adjOwner == player_name):
                x[0][n][surrAreaN] = 1

            else:
                m = (adjOwner + 4 - player_name) % 4
                x[m+1][n][surrAreaN] = probability_of_successful_attack(board, pArea, surrArea)


    # global parameters:
    # TODO: get parameters
    paramOut = torch.zeros(12)

    return torch.cat([x.reshape(-1), paramOut])




    



        
if __name__=="__main__":
    x = torch.zeros((2,3,3))
    exit(0)