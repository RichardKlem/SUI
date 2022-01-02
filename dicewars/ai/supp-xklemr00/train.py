# must be run from the SUI directory

from random import random, shuffle
from math import floor

import os, pickle, torch, copy

NGAMES = 10

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
    shuffle(opponents)
    opp1 = opponents[0]
    opp2 = opponents[1]
    opp3 = opponents[2]

    # run the game
    command = "python3 ./scripts/dicewars-ai-only.py -r -n 1 -l ../logs --ai " + \
        opp1 + " " + \
        opp2 + " " + \
        opp3 + " xklemr00"
    os.system(command)
    print("Battles complete!")

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


            # too early to evaluate
            if refTurn < 2: 
                i2 += 1
                continue

            # increment to make turn gap
            if refTurn - 2 < currTurn:
                i2 += 1
                # too late to evaluate
                if i2 >= len(results):
                    break
                continue

            currDice = line1[3]
            refDice  = line2[3]

            if refDice > currDice:
                # you're doing great!
                i1 += 1
                continue

            # time to train
            input = line1[0]
            result = model.forward(input)
            outputs.append(result)

            i1 += 1

        print("Turns to evaluate:", len(outputs))

        for out in outputs:
            output = copy.copy(out)
            expected = (1.0-output)/5.0
            loss = criterion(output, expected) + 1000**-(700*(output-1/6)**2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.save("dicewars/ai/xklemr00/model.pth")

        








