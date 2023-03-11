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


def coups_possibles(positions: dict, couleur=0):
    """
    Cette fonction retourne tous les coups possibles en fonctions des positions des autres pions.
    :param couleur: Couleurs du joueur du tour
    :param positions: Dictionnaire avec la position d'un pion en clé et avec les informations du pion en valeurs.
    :return: Liste des coups possibles.
    """
    coups = []
    # Avancer
    if len(coups) == 0:
        for case in positions.keys():
            # Pions noirs
            if positions[case][0] and couleur == 1:
                # Pions sur la rangée de droite et sur une ligne paire
                # Pions sur la rangée de gauche et sur une ligne impaire
                if (case // 5) % 2 == 1 and case % 5 == 0 or (case // 5) % 2 == 1 and case % 5 == 1:
                    if case + 5 not in positions: coups.append(f'{case}-{case + 5}')
                # Tous les autres pions
                else:
                    parity = (case // 5) % 2
                    if parity == 0 and case % 5 == 0:
                        parity = 1
                    if case + 5 - parity not in positions:
                        coups.append(f'{case}-{case + 5 - parity}')
                    if case + 6 - parity not in positions:
                        coups.append(f'{case}-{case + 6 - parity}')
            # Pions blancs
            elif positions[case][0] == 0 and couleur == 0:
                # Pions sur la rangée de droite et sur une ligne paire
                # Pions sur la rangée de gauche et sur une ligne impaire
                if (case // 5) % 2 == 1 and case % 5 == 0 or (case // 5) % 2 == 1 and case % 5 == 1:
                    if case - 5 not in positions: coups.append(f'{case}-{case - 5}')
                # Tous les autres pions
                else:
                    parity = (case // 5) % 2
                    if parity == 0 and case % 5 == 0:
                        parity = 1
                    if case - 5 - parity not in positions:
                        coups.append(f'{case}-{case - 5 - parity}')
                    if case - 4 - parity not in positions:
                        coups.append(f'{case}-{case - 4 - parity}')
    return coups
