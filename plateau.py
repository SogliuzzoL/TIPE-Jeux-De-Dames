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
            self.pions.append(Pion(i, 0))

        for i in range(31, 51):
            self.pions.append(Pion(i, 1))

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
    for position in positions.keys():
        # Si le pion est blanc
        if not positions[position][0] and not couleur:
            # Coup pour un point
            if not positions[position][1]:
                # Devant droite et gauche
                if not (position + 5 in positions) and position // 5:
                    coups.append(f"{position}-{position + 5}")
                if not (position + 6 in positions) and position % 5:
                    coups.append(f"{position}-{position + 6}")

        # Si le pion est noir
        if positions[position][0] and couleur:
            # Coup pour un point
            if not positions[position][1]:
                # Devant droite et gauche
                if not (position - 5 in positions) and position // 5 - 1:
                    coups.append(f"{position}-{position - 5}")
                if not (position - 6 in positions) and position % 5 - 1:
                    coups.append(f"{position}-{position - 6}")
    return coups
