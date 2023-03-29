import time

import treelib.exceptions
from treelib import Tree


class Pion:
    def __init__(self, emplacement, couleur):
        """
        Cette fonction permet de créer un pion sur le plateau.
        :param x: Représente la coordonnée en x (entre 0 et 9) du pion.
        :param y: Représente la coordonnée en y (entre 0 et 9) du pion.
        :param couleur: Pion de couleur Blanc (0) ou Noir (1).
        """
        self.emplacement = emplacement
        self.color = couleur
        self.dame = False


class Plateau:
    def __init__(self):
        self.pions = []
        for i in range(1, 21):
            self.pions.append(Pion(i, 1))

        for i in range(31, 51):
            self.pions.append(Pion(i, 0))

    def positions(self):
        positions = {}
        for pion in self.pions:
            positions[pion.emplacement] = [pion.color, pion.dame]
        return positions


def coups_prises_pions(case: int, positions: dict, couleur=0, tree=None, parent=None):
    if tree is None:
        tree = Tree()
        tree.create_node(case, case)
        parent = case
    parity = ((case - 1) // 5) % 2
    datas = []
    if parity:
        datas = [[5, 11, 0], [4, 9, -1], [-5, -9, 0], [-6, -11, -1]]
    else:
        datas = [[6, 11, 0], [5, 9, -1], [-4, -9, 0], [-5, -11, -1]]
    for i in range(len(datas)):
        if case + datas[i][0] in positions.keys() and case + datas[i][1] not in positions.keys() and (
                case + datas[i][2]) % 5 != 0:
            if positions[case + datas[i][0]][0] != couleur and 0 < case + datas[i][0] < 50:
                positions_copie = positions.copy()
                data = positions_copie[case]
                del positions_copie[case + datas[i][0]]
                positions_copie[case + datas[i][1]] = data
                try:
                    tree.create_node(f'{case}x{case + datas[i][1]}', f'{case}x{case + datas[i][1]}', parent=parent)
                except treelib.exceptions.DuplicatedNodeIdError:
                    tree.create_node(f'{case}x{case + datas[i][1]}-{time.time()}',
                                     f'{case}x{case + datas[i][1]}-{time.time()}', parent=parent)
                coups_prises_pions(case + datas[i][1], positions_copie, couleur, tree, f'{case}x{case + datas[i][1]}')
    return tree


def coups_avancer_pions(case: int, positions: dict, couleur=0):
    coups = []
    modifier_couleur = 1
    modifier_couleur2 = 0
    parity = ((case - 1) // 5) % 2
    if couleur == 0:
        modifier_couleur = -1
        if parity: modifier_couleur2 = -1
        else: modifier_couleur2 = 1
    if (case // 5) % 2 == 1 and case % 5 == 0 or (case // 5) % 2 == 1 and case % 5 == 1:
        if case + 5 * modifier_couleur not in positions: coups.append(f'{case}-{case + 5 * modifier_couleur}')
    else:
        if case + (5 - parity) * modifier_couleur + modifier_couleur2 not in positions:
            coups.append(f'{case}-{case + (5 - parity) * modifier_couleur + modifier_couleur2}')
        if case + (6 - parity) * modifier_couleur + modifier_couleur2 not in positions:
            coups.append(f'{case}-{case + (6 - parity) * modifier_couleur + modifier_couleur2}')
    return coups


def coups_avancer_dames(case, positions, couleur):
    coups = []

    return coups


def coups_possibles(positions: dict, couleur=0):
    """
    Cette fonction retourne tous les coups possibles en fonctions des positions des autres pions.
    :param couleur: Couleurs du joueur du tour
    :param positions: Dictionnaire avec la position d'un pion en clé et avec les informations du pion en valeurs.
    :return: Liste des coups possibles.
    """
    coups = []
    coups_tempo = []
    max_coups = 0
    # Prises
    for case in positions.keys():
        if positions[case][0] == couleur:
            if not positions[case][1]:
                tree = coups_prises_pions(case, positions, couleur)
                if tree.depth():
                    tree.show()
                    paths = tree.paths_to_leaves()
                    for path in paths:
                        if max_coups < len(path):
                            max_coups = len(path)
                        coups_tempo.append(path)
    # Trie des coups pour prises
    for i in range(len(coups_tempo)):
        if len(coups_tempo[i]) == max_coups:
            for j in range(len(coups_tempo[i])):
                if '-' in str(coups_tempo[i][j]):
                    split = str(coups_tempo[i][j]).split('-')
                    coups_tempo[i][j] = split[0]
            coups.append(coups_tempo[i])
    # Avancer
    if len(coups) == 0:
        for case in positions.keys():
            if not positions[case][1] and positions[case][0] == couleur:
                coups.extend(coups_avancer_pions(case, positions, couleur))
            elif positions[case][1] and positions[case][0] == couleur:
                coups.extend(coups_avancer_dames(case, positions, couleur))
    return coups
