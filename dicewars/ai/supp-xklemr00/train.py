# must be run from the SUI directory

from random import random
from math import floor

import os, pickle, torch

NGAMES = 1

import dicewars.ai.xklemr00.dggraphnet

opponents = [
    "dt.sdc",
    "dt.rand",
    "dt.ste",
    "dt.stei",
    "kb.sdc_at",
    "kb.xlogin42"
]

model = dicewars.ai.xklemr00.dggraphnet.DGGraphNet(4*34*34+12, 34, 12, 12, 6)
model.load("dicewars/ai/xklemr00/model.pth")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# run games
for i in range(NGAMES):

    # select random opponents
    opp1 = opponents[floor(random()*len(opponents))]
    opp2 = opponents[floor(random()*len(opponents))]
    opp3 = opponents[floor(random()*len(opponents))]

    # run the game
    command = "python3 ./scripts/dicewars-ai-only.py -r -b 11 -o 22 -s 33 -c 44 -n 1 -l ../logs --ai " + \
        opp1 + " " + \
        opp2 + " " + \
        opp3 + " xklemr00"
    os.system(command)

    # evaluate the results
    with open("game_records.pkl", "rb") as f:
        results = pickle.load(f)

        if len(results) < 2: continue

        i1 = 0
        i2 = 1

        batch_counter = 0

        outputs = []

        # results structure:
        # 0 - NN input
        # 1 - NN output
        # 2 - turn number
        # 3 - dice count that turn

        while True:
            line1 = results[i1]
            line2 = results[i2]

            currTurn = line1[2]
            refTurn = line2[2]

            print(currTurn, refTurn)

            # too early to evaluate
            if refTurn < 2: 
                i2 += 1
                print("i2 increment:", i2)
                continue

            # increment to make turn gap
            if refTurn - 2 < currTurn:
                i2 += 1
                print("i2 increment:", i2)
                # too late to evaluate
                if i2 >= len(results):
                    break
                continue

            currDice = line1[3]
            refDice  = line2[3]

            if refDice > currDice:
                # you're doing great!
                i1 += 1
                print("i1 increment:", i1)
                continue

            # time to train
            input = line1[0]
            result = model.forward(input)
            outputs.append(result)

            i1 += 1
            print("i1 increment:", i1)


        for output in outputs:
            print(output)
            expected = (1.0-output)/5.0
            loss = criterion(output, expected) + 1000**-(700*(output-1/6)**2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        








