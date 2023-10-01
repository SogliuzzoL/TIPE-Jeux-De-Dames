import datetime
import random
import time

import numpy
import numpy as np
import torch
from plateau import Plateau, coups_possibles
from torch import Tensor
from torch.nn import Module, Linear, Softsign, Tanh

# Get cpu, gpu or mps device for training.
print(f"Cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")

torch.set_default_device(device)

input_layer_len = 57


class Model(Module):
    def __init__(self, n_inputs):
        """
        :param n_inputs: Nombre d'entrées que le model aura
        """
        super(Model, self).__init__()
        self.hidden1 = Linear(n_inputs, 7*n_inputs)
        self.act1 = Tanh()
        self.hidden2 = Linear(7*n_inputs, 7*n_inputs)
        self.act2 = Softsign()
        self.hidden3 = Linear(7*n_inputs, 7*n_inputs)
        self.act3 = Tanh()
        self.hidden4 = Linear(7*n_inputs, 50)
        self.act4 = Softsign()

    def forward(self, data):
        """
        :param data: Données d'entrées servant à calculer les données de sorties
        :return: Renvoie les données de sorties
        """
        data = self.hidden1(data)
        data = self.act1(data)
        data = self.hidden2(data)
        data = self.act2(data)
        data = self.hidden3(data)
        data = self.act3(data)
        data = self.hidden4(data)
        data = self.act4(data)
        return data


def predict_ia(row, model) -> numpy.ndarray:
    """
    :param row: Liste des données d'entrée
    :param model: Model à utiliser
    :return: Renvoie une liste contenant la sortie du model
    """
    predicted_datas = model(Tensor(row))
    predicted_datas = predicted_datas.detach().numpy()
    return predicted_datas


def run_ia(plateau, model_start_case: Model, model_end_case: Model) -> str:
    """
    :param plateau: Plateau de jeu
    :param model_start_case: Model permettant de choisir la case de départ
    :param model_end_case: Model permettant de choisir la case d'arrivée
    :return: Une chaîne de caractère contenant le coup à jouer
    """
    positions = plateau.positions()
    informations = plateau.plateau_information()
    row = [0 for _ in range(input_layer_len)]
    for i in range(input_layer_len):
        if i == 0:
            if plateau.round_side == 1:
                row[i] = -1
            else:
                row[i] = 1
        elif i in positions:
            if positions[i][0] == 0:
                if not positions[i][1]:
                    row[i] = 0.5
                else:
                    row[i] = 1
            else:
                if not positions[i][1]:
                    row[i] = -0.5
                else:
                    row[i] = -1

    row[51] = informations['compte_noirs']
    row[52] = informations['compte_blancs']
    row[53] = informations['dames_noirs']
    row[54] = informations['dames_blancs']
    row[55] = informations['centre_noirs']
    row[56] = informations['centre_blancs']

    y_start = predict_ia(row, model_start_case)
    y_end = predict_ia(row, model_end_case)
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
        if i + 1 in coup_dict and abs(y_start[i]) > abs(y_best_start[1]):
            y_best_start = (i + 1, y_start[i])
    y_best_end = (0, 0)
    for i in range(50):
        if coup_dict != {}:
            if i + 1 in coup_dict[y_best_start[0]] and abs(y_end[i]) > abs(y_best_end[1]):
                y_best_end = (i + 1, y_end[i])
    real_coup = ''
    for coup in coups:
        if coup.startswith(str(y_best_start[0])) and coup.endswith(str(y_best_end[0])):
            if (len(str(y_best_end[0])) == 1 and (coup[-2] == 'x' or coup[-2] == '-')) or (
                    len(str(y_best_end[0])) == 2 and (coup[-3] == 'x' or coup[-3] == '-')):
                real_coup = coup
                break
    return real_coup


def simulation_ia(player_white: tuple, player_black: tuple) -> int:
    """
    :param player_white: Tuple contenant (model_start, model_end) pour le joueur blanc
    :param player_black: Tuple contenant (model_start, model_end) pour le joueur noir
    :return: Renvoie 0 si les blancs ont gagné, 1 si les noirs ont gagné et 2 s'il y a égalité
    """
    plateau = Plateau()
    while plateau.check_win() == -1:
        if plateau.round_side == 0:
            plateau.jouer_coup(run_ia(plateau, player_white[0], player_white[1]), 0)
            plateau.round_side = 1
        else:
            plateau.jouer_coup(run_ia(plateau, player_black[0], player_black[1]), 1)
            plateau.round_side = 0
    win = plateau.check_win()
    return win


def mutation(model_a: Model, model_b: Model, rate: float, percent: float):
    """
    :param model_a: Model à muter
    :param model_b: Model servant à la mutation
    :param rate: Ratio d'emprunt génétique du model A au model B
    :param percent: Pourcentage d'une mutation aléatoire
    :return: Ne retourne rien
    """
    # Bias Mutation
    bias_list_a = [model_a.hidden1.bias.data, model_a.hidden2.bias.data, model_a.hidden3.bias.data,
                   model_a.hidden4.bias.data]
    bias_list_b = [model_b.hidden1.bias.data, model_b.hidden2.bias.data, model_b.hidden3.bias.data,
                   model_b.hidden4.bias.data]
    for i in range(len(bias_list_a)):
        len_bias = len(bias_list_a[i])
        rating = [random.randint(0, len_bias - 1) for _ in range(int(len_bias*rate))]
        percentage = [random.randint(0, len_bias - 1) for _ in range(int(len_bias*percent))]
        for r in rating:
            bias_list_a[i][r] = bias_list_b[i][r]
        for p in percentage:
            bias_list_a[i][p] = random.randint(-999_999, 999_999) / 1_000_000
    # Weights mutation
    weights_list_a = [model_a.hidden1.weight.data, model_a.hidden2.weight.data, model_a.hidden3.weight.data,
                      model_a.hidden4.weight.data]
    weights_list_b = [model_b.hidden1.weight.data, model_b.hidden2.weight.data, model_b.hidden3.weight.data,
                      model_b.hidden4.weight.data]
    for i in range(len(weights_list_a)):
        for j in range(len(weights_list_a[i])):
            len_weights = len(weights_list_a[i][j])
            rating = [random.randint(0, len_weights - 1) for _ in range(int(len_weights*rate))]
            percentage = [random.randint(0, len_weights - 1) for _ in range(int(len_weights*percent))]
            for r in rating:
                weights_list_a[i][j][r] = weights_list_b[i][j][r]
            for p in percentage:
                weights_list_a[i][j][p] = random.randint(-999_999, 999_999) / 1_000_000


def training(model_start: list, model_end: list, n_gen: int) -> (list, list, list):
    """
    :param model_start: Liste des model de départ
    :param model_end: Liste des model d'arrivée
    :param n_gen: Nombre de générations à faire
    :return: Renvoie trois listes sous la forme (list_model_start, list_model_end, list_result_game)
    """
    t0 = time.time()
    results = []
    for gen in range(n_gen):
        results = []
        next_model_start = []
        next_model_end = []
        for i in range(1, len(model_start), 2):
            result = simulation_ia((model_start[i], model_end[i]), (model_start[i - 1], model_end[i - 1]))
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
            t1 = time.time()
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Génération {gen + 1}. Moyenne de temps:{(t1-t0)/(gen + 1)}')
        model_start = next_model_start
        model_end = next_model_end
    return model_start, model_end, results


def start_training(model_start_load=None, model_end_load=None) -> (Model, Model):
    """
    :param model_start_load: Mettre un model de départ en cas d'upgrade de celui-ci
    :param model_end_load: Mettre un model d'arriver en cas d'upgrade de celui-ci
    :return: Renvoie deux model sous le format suivant: (best_model_start, best_model_end)
    """
    expo = 1
    gen_mul = 1_000
    if not (model_start_load is None or model_end_load is None):
        print("Upgrade actual model")
        model_start = [model_start_load for _ in range(2 ** expo)]
        model_end = [model_end_load for _ in range(2 ** expo)]
    else:
        model_start = [Model(input_layer_len) for _ in range(2 ** expo)]
        model_end = [Model(input_layer_len) for _ in range(2 ** expo)]
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


def load_model() -> (Model, Model):
    """
    :return: Renvoie deux models sous le format suivant: (model_loaded_start, model_loaded_end)
    """
    model_start = Model(input_layer_len)
    model_start.load_state_dict(torch.load('model_start'))
    model_start.eval()
    model_end = Model(input_layer_len)
    model_end.load_state_dict(torch.load('model_end'))
    model_end.eval()
    return model_start, model_end
