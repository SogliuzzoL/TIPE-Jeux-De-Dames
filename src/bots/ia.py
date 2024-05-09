import copy
import datetime
import random
import time

import numpy
import torch
from plateau import Plateau, coups_possibles
from torch import Tensor
from torch.nn import Module, Linear, Softsign

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
        self.hidden1 = Linear(n_inputs, 2 * n_inputs)
        self.act1 = Softsign()
        self.hidden2 = Linear(2 * n_inputs, 2 * n_inputs)
        self.act2 = Softsign()
        self.hidden3 = Linear(2 * n_inputs, 2 * n_inputs)
        self.act3 = Softsign()
        self.hidden4 = Linear(2 * n_inputs, 50)
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


def simulation_ia_vs_ia(player_white: tuple, player_black: tuple) -> tuple[int, dict]:
    """
    :param player_white: Tuple contenant (model_start, model_end) pour le joueur blanc
    :param player_black: Tuple contenant (model_start, model_end) pour le joueur noir
    :return: Renvoie 0 si les blancs ont gagné, 1 si les noirs ont gagné et 2 s'il y a égalité et les informations du plateau
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
    return win, plateau.plateau_information()


def simulation_ia_vs_montecarlo(model: tuple, model_color=0) -> tuple[int, dict]:
    """
    :param model_color: couleur du model (0=white, 1=black)
    :param model: Tuple contenant (model_start, model_end)
    :return: Renvoie 0 si les blancs ont gagné, 1 si les noirs ont gagné et 2 s'il y a égalité et les informations du plateau
    """
    plateau = Plateau()
    while plateau.check_win() == -1:
        if plateau.round_side == model_color:
            plateau.jouer_coup(run_ia(plateau, model[0], model[1]), model_color)
            plateau.round_side = 1 - model_color
        else:
            coups = coups_possibles(plateau.positions(), plateau.round_side)
            plateau.jouer_coup(coups[random.randint(0, len(coups) - 1)], 1 - model_color)
            plateau.round_side = model_color
    win = plateau.check_win()
    return win, plateau.plateau_information()


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
        rating = [random.randint(0, len_bias - 1) for _ in range(int(len_bias * rate))]
        percentage = [random.randint(0, len_bias - 1) for _ in range(int(len_bias * percent))]
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
            rating = [random.randint(0, len_weights - 1) for _ in range(int(len_weights * rate))]
            percentage = [random.randint(0, len_weights - 1) for _ in range(int(len_weights * percent))]
            for r in rating:
                weights_list_a[i][j][r] = weights_list_b[i][j][r]
            for p in percentage:
                weights_list_a[i][j][p] = random.randint(-999_999, 999_999) / 1_000_000


def training(model_start_blanc: list, model_end_blanc: list, model_start_noir: list, model_end_noir: list, n_gen: int,
             bot_id=0) -> (Model, Model, Model, Model):
    """
    :param model_start_blanc: Liste des model de départ blanc
    :param model_end_blanc: Liste des model d'arrivée blanc
    :param model_start_noir: Liste des model de départ noir
    :param model_end_noir: Liste des model d'arrivée noir
    :param n_gen: Nombre de générations à faire
    :param bot_id: Choix de l'algorithme qui va affronter les models (0 = Monte-Carlo)
    :return: Renvoie quatre models sous la forme suivante (best_start_blanc, best_end_blanc, best_start_noir, best_end_noir)
    """
    t0 = time.time()
    score_blancs_plus_id_model = []
    score_noirs_plus_id_model = []
    n = len(model_start_blanc)
    best_start_blanc, best_end_blanc, best_start_noir, best_end_noir = None, None, None, None
    for gen in range(n_gen):
        # Simulation des parties
        score_blancs = []
        score_noirs = []
        for i in range(n):
            # Simulation blancs
            result = simulation_ia_vs_montecarlo((model_start_blanc[i], model_end_blanc[i]), 0)
            score = result[1]['compte_blancs'] - result[1]['compte_noirs']
            if result[0] == 0:
                score += 20
            elif result[0] == 1:
                score -= 20
            score_blancs.append(score)

            # Simulation noirs
            result = simulation_ia_vs_montecarlo((model_start_noir[i], model_end_noir[i]), 1)
            score = result[1]['compte_noirs'] - result[1]['compte_blancs']
            if result[0] == 0:
                score -= 20
            elif result[0] == 1:
                score += 20
            score_noirs.append(score)

        # Trie des meilleurs models
        score_blancs_plus_id_model = [(score_blancs[i], i) for i in range(len(score_blancs))]
        score_blancs_plus_id_model.sort(key=lambda x: x[0], reverse=True)

        score_noirs_plus_id_model = [(score_noirs[i], i) for i in range(len(score_noirs))]
        score_noirs_plus_id_model.sort(key=lambda x: x[0], reverse=True)

        # Création de la nouvelle génération avec 1/4 des meilleurs models
        new_start_noir = []
        new_end_noir = []
        new_start_blanc = []
        new_end_blanc = []
        for i in range(int(n / 3)):
            new_start_blanc.append(model_start_blanc[score_blancs_plus_id_model[i][1]])
            new_end_blanc.append(model_start_blanc[score_blancs_plus_id_model[i][1]])
            new_start_noir.append(model_start_noir[score_noirs_plus_id_model[i][1]])
            new_end_noir.append(model_start_noir[score_noirs_plus_id_model[i][1]])

        # Mutation du meilleur model
        for i in range(int(n / 3)):
            copy_meilleur_blanc_start = copy.deepcopy(new_start_blanc[0])
            mutation(copy_meilleur_blanc_start, new_start_blanc[i], 0.5, 0.05)
            new_start_blanc.append(copy_meilleur_blanc_start)

            copy_meilleur_blanc_end = copy.deepcopy(new_end_blanc[0])
            mutation(copy_meilleur_blanc_end, new_end_blanc[i], 0.5, 0.05)
            new_end_blanc.append(copy_meilleur_blanc_end)

            copy_meilleur_noir_start = copy.deepcopy(new_start_noir[0])
            mutation(copy_meilleur_noir_start, new_start_noir[i], 0.5, 0.05)
            new_start_noir.append(copy_meilleur_noir_start)

            copy_meilleur_noir_end = copy.deepcopy(new_end_noir[0])
            mutation(copy_meilleur_noir_end, new_end_noir[i], 0.5, 0.05)
            new_end_noir.append(copy_meilleur_noir_end)

        # Ajout de nouveaux models
        for i in range(n - 2 * int(n / 3)):
            new_start_blanc.append(Model(input_layer_len))
            new_end_blanc.append(Model(input_layer_len))
            new_start_noir.append(Model(input_layer_len))
            new_end_noir.append(Model(input_layer_len))

        # Modification des listes de models pour la prochaine génération
        model_start_blanc = new_start_blanc
        model_end_blanc = new_end_blanc
        model_start_noir = new_start_noir
        model_end_noir = new_end_noir

        best_start_blanc, best_end_blanc, best_start_noir, best_end_noir = model_start_blanc[
            score_blancs_plus_id_model[0][1]], model_end_blanc[score_blancs_plus_id_model[0][1]], model_start_noir[
            score_noirs_plus_id_model[0][1]], model_end_noir[score_noirs_plus_id_model[0][1]]

        if (gen + 1) % 10 == 0:
            t1 = time.time()
            print(
                f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Génération {gen + 1}. Moyenne de temps '
                f'pour une génération:{(t1 - t0) / (gen + 1)}')
            print(
                f"{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Score Models Blancs: {score_blancs_plus_id_model}")
            # Sauvegarde du meilleur model
            torch.save(best_start_blanc.state_dict(),
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_gen{gen + 1}_" + 'model_start_blanc')
            torch.save(best_end_blanc.state_dict(),
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_gen{gen + 1}_" + 'model_end_blanc')
            torch.save(best_start_noir.state_dict(),
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_gen{gen + 1}_" + 'model_start_noir')
            torch.save(best_end_noir.state_dict(),
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_gen{gen + 1}_" + 'model_end_noir')

    return best_start_blanc, best_end_blanc, best_start_noir, best_end_noir


def start_training(model_start_load_blanc=None, model_end_load_blanc=None, model_start_load_noir=None,
                   model_end_load_noir=None) -> (Model, Model, Model, Model):
    """
    :param model_start_load_blanc: Mettre un model de départ blanc en cas d'upgrade de celui-ci
    :param model_end_load_blanc: Mettre un model d'arriver blanc en cas d'upgrade de celui-ci
    :param model_start_load_noir: Mettre un model de départ noir en cas d'upgrade de celui-ci
    :param model_end_load_noir: Mettre un model d'arriver noir en cas d'upgrade de celui-ci
    :return: Renvoie deux model sous le format suivant: (best_model_start_blanc, best_model_end_blanc, best_model_start_noir, best_model_end_noir)
    """
    gen = 10_000
    if not (
            model_start_load_blanc is None or model_end_load_blanc is None or model_start_load_noir is None or model_end_load_noir is None):
        print("Upgrade actual model")
        model_start_blanc = [model_start_load_blanc for _ in range(20)]
        model_end_blanc = [model_end_load_blanc for _ in range(20)]
        model_start_noir = [model_start_load_noir for _ in range(20)]
        model_end_noir = [model_end_load_noir for _ in range(20)]
        model_start_blanc += [Model(input_layer_len) for _ in range(80)]
        model_end_blanc += [Model(input_layer_len) for _ in range(80)]
        model_start_noir += [Model(input_layer_len) for _ in range(80)]
        model_end_noir += [Model(input_layer_len) for _ in range(80)]
    else:
        model_start_blanc = [Model(input_layer_len) for _ in range(100)]
        model_end_blanc = [Model(input_layer_len) for _ in range(100)]
        model_start_noir = [Model(input_layer_len) for _ in range(100)]
        model_end_noir = [Model(input_layer_len) for _ in range(100)]
    print(
        f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Nouveau training avec {len(model_start_blanc)} models')
    model_start_blanc, model_end_blanc, model_start_noir, model_end_noir = training(model_start_blanc, model_end_blanc,
                                                                                    model_start_noir, model_end_noir,
                                                                                    gen)
    torch.save(model_start_blanc.state_dict(), 'model_start_blanc')
    torch.save(model_end_blanc.state_dict(), 'model_end_blanc')
    torch.save(model_start_noir.state_dict(), 'model_start_noir')
    torch.save(model_end_noir.state_dict(), 'model_end_noir')
    return model_start_blanc, model_end_blanc, model_start_noir, model_end_noir


def load_model() -> (Model, Model, Model, Model):
    """
    :return: Renvoie quatre models sous le format suivant: (model_loaded_start_blanc, model_loaded_end_blanc, model_loaded_start_noir, model_loaded_end_noir)
    """
    model_start_blanc = Model(input_layer_len)
    model_start_blanc.load_state_dict(torch.load('model_start_blanc'))
    model_start_blanc.eval()
    model_end_blanc = Model(input_layer_len)
    model_end_blanc.load_state_dict(torch.load('model_end_blanc'))
    model_end_blanc.eval()
    model_start_noir = Model(input_layer_len)
    model_start_noir.load_state_dict(torch.load('model_start_noir'))
    model_start_noir.eval()
    model_end_noir = Model(input_layer_len)
    model_end_noir.load_state_dict(torch.load('model_end_noir'))
    model_end_noir.eval()
    return model_start_blanc, model_end_blanc, model_start_noir, model_end_noir
