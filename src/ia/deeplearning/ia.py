import math
import random

import torch
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.w1 = torch.randn(150, 10) / math.sqrt(150)
        self.w1.requires_grad_()
        self.b1 = torch.zeros(10, requires_grad=True)
        self.w2 = torch.randn(10, 50) / math.sqrt(50)
        self.w2.requires_grad_()
        self.b2 = torch.zeros(50, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb, network):
    hidden_layer = log_softmax(xb.T @ network.w1 + network.b1)
    output_layer = log_softmax(hidden_layer @ network.w2 + network.b2)
    return output_layer


def make_prediction(plateau, network: NeuralNetwork):
    # Creation de l'input layer
    inputs = [[0, 0, 0] for i in range(1, 51)]
    for pion in plateau.pions:
        inputs[pion.emplacement - 1] = [1, pion.color, int(pion.dame)]
    input_layer = torch.zeros(150, 1)
    n = 0
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            input_layer[n] = inputs[i][j]
            n += 1
    print(model(input_layer, network))
