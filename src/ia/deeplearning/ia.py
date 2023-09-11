import datetime
import random

import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Sigmoid
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.utils.tensorboard import SummaryWriter

from plateau import Plateau, coups_possibles

# Get cpu, gpu or mps device for training.
print(f"Cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")

torch.set_default_device(device)


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, n_inputs)
        kaiming_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(n_inputs, n_inputs // 2)
        kaiming_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(n_inputs // 2, n_inputs // 2)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        # third hidden layer and output
        self.hidden4 = Linear(n_inputs // 2, 50)
        xavier_uniform_(self.hidden3.weight)
        self.act4 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer and output
        X = self.hidden4(X)
        X = self.act4(X)
        return X


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


def run(plateau, model_start_case: MLP, model_end_case: MLP):
    positions = plateau.positions()
    row = [0.5 for _ in range(51)]
    for i in range(51):
        if i == 0:
            row[i] = plateau.round_side
        elif i in positions:
            row[i] = positions[i][0]

    y_start = predict(row, model_start_case)
    y_end = predict(row, model_end_case)
    coups = coups_possibles(positions, plateau.round_side)

    coup_dict = {}
    for coup in coups:
        temp_coup = []
        if '-' in coup:
            temp_coup = coup.split('-')
        elif 'x' in coup:
            temp_coup = coup.split('x')
        if int(temp_coup[0]) not in coup_dict:
            coup_dict[int(temp_coup[0])] = []
        coup_dict[int(temp_coup[0])].append(int(temp_coup[-1]))

    y_best_start = (0, 0)
    for i in range(50):
        if i + 1 in coup_dict and y_start[0][i] > y_best_start[1]:
            y_best_start = (i + 1, y_start[0][i])
    y_best_end = (0, 0)
    for i in range(50):
        if i + 1 in coup_dict[y_best_start[0]] and y_end[0][i] > y_best_end[1]:
            y_best_end = (i + 1, y_end[0][i])
    real_coup = ''
    for coup in coups:
        if coup.startswith(str(y_best_start[0])) and coup.endswith(str(y_best_end[0])):
            real_coup = coup
            break
    return real_coup


def simu(player_white: tuple, player_black: tuple) -> int:
    plateau = Plateau()
    while plateau.check_win() == -1:
        if plateau.round_side == 0:
            plateau.jouer_coup(run(plateau, player_white[0], player_white[1]), 0)
            plateau.round_side = 1
        else:
            plateau.jouer_coup(run(plateau, player_black[0], player_black[1]), 1)
            plateau.round_side = 0
    win = plateau.check_win()
    return win


def mutation(model_a: MLP, model_b: MLP, rate: float, percent: float):
    for i in range(len(model_a.hidden1.weight.data)):
        for j in range(len(model_a.hidden1.weight.data[i])):
            if random.random() < rate:
                model_a.hidden1.weight.data[i][j] = model_b.hidden1.weight.data[i][j]
            if random.random() < percent:
                model_a.hidden1.weight.data[i][j] = torch.rand((1, 1))
    for i in range(len(model_a.hidden2.weight.data)):
        for j in range(len(model_a.hidden2.weight.data[i])):
            if random.random() < rate:
                model_a.hidden2.weight.data[i][j] = model_b.hidden2.weight.data[i][j]
            if random.random() < percent:
                model_a.hidden2.weight.data[i][j] = torch.rand((1, 1))
    for i in range(len(model_a.hidden3.weight.data)):
        for j in range(len(model_a.hidden3.weight.data[i])):
            if random.random() < rate:
                model_a.hidden3.weight.data[i][j] = model_b.hidden3.weight.data[i][j]
            if random.random() < percent:
                model_a.hidden3.weight.data[i][j] = torch.rand((1, 1))
    for i in range(len(model_a.hidden4.weight.data)):
        for j in range(len(model_a.hidden4.weight.data[i])):
            if random.random() < rate:
                model_a.hidden4.weight.data[i][j] = model_b.hidden4.weight.data[i][j]
            if random.random() < percent:
                model_a.hidden4.weight.data[i][j] = torch.rand((1, 1))


def training(model_start, model_end, n_gen):
    results = []
    for gen in range(n_gen):
        results = []
        next_model_start = []
        next_model_end = []
        for i in range(1, len(model_start), 2):
            result = simu((model_start[i], model_end[i]), (model_start[i - 1], model_end[i - 1]))
            if result == 0:
                temp_start = model_start[i]
                mutation(model_start[i], model_start[i - 1], 0, 0.01)
                mutation(temp_start, model_start[i - 1], 0.1, 0.05)
                next_model_start.append(model_start[i])
                next_model_start.append(temp_start)

                temp_end = model_end[i]
                mutation(model_end[i], model_end[i - 1], 0, 0.01)
                mutation(temp_end, model_end[i - 1], 0.1, 0.05)
                next_model_end.append(model_end[i])
                next_model_end.append(temp_end)

            elif result == 1:
                temp_start = model_start[i - 1]
                mutation(model_start[i - 1], model_start[i], 0, 0.01)
                mutation(temp_start, model_start[i], 0.1, 0.05)
                next_model_start.append(model_start[i - 1])
                next_model_start.append(temp_start)

                temp_end = model_end[i - 1]
                mutation(model_end[i - 1], model_end[i], 0, 0.01)
                mutation(temp_end, model_end[i], 0.1, 0.05)
                next_model_end.append(model_end[i - 1])
                next_model_end.append(temp_end)

            else:
                temp_start = model_start[i]
                mutation(model_start[i], model_start[i - 1], 0, 0.05)
                mutation(model_start[i - 1], temp_start, 0, 0.05)
                next_model_start.append(model_start[i])
                next_model_start.append(model_start[i - 1])

                temp_end = model_end[i]
                mutation(model_end[i], model_end[i - 1], 0, 0.05)
                mutation(model_end[i - 1], temp_end, 0, 0.05)
                next_model_end.append(model_end[i])
                next_model_end.append(model_end[i - 1])

            results.append(result)
        if (gen + 1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Génération {gen + 1}')
        model_start = next_model_start
        model_end = next_model_end
    return model_start, model_end, results


def start_training(model_start_load=None, model_end_load=None):
    expo = 3
    gen_mul = 100_000
    """if True:
        expo = 1
        gen_mul = 200"""
    if model_start_load is None and model_end_load is None:
        model_start = [model_start_load for _ in range(2 ** expo)]
        model_end = [model_end_load for _ in range(2 ** expo)]
    else:
        model_start = [MLP(51) for _ in range(2 ** expo)]
        model_end = [MLP(51) for _ in range(2 ** expo)]
    n = 1
    while len(model_start) > 1:
        print(
            f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Nouveau training avec {len(model_start)} models')
        model_start, model_end, results = training(model_start, model_end, n * gen_mul)
        new_start = []
        new_end = []
        for i in range(len(results)):
            if results[i] == 0:
                new_start.append(model_start[2 * i])
                new_end.append(model_end[2 * i])
            elif results[i] == 1:
                new_start.append(model_start[2 * i + 1])
                new_end.append(model_end[2 * i + 1])
            else:
                new_start.append(model_start[2 * i])
                new_end.append(model_end[2 * i])
        model_start = new_start
        model_end = new_end
        model_start.reverse()
        model_end.reverse()
        n += 1
        torch.save(model_start[0].state_dict(), 'model_start')
        torch.save(model_end[0].state_dict(), 'model_end')
    return model_start[0], model_end[0]


def load_model():
    model_start = MLP(51)
    model_start.load_state_dict(torch.load('model_start'))
    model_start.eval()
    model_end = MLP(51)
    model_end.load_state_dict(torch.load('model_end'))
    model_end.eval()
    return model_start, model_end
