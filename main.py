class Pion:
    def __init__(self, x, y, color):
        """
        Cette fonction permet de créer un pion sur le plateau
        :param x: Représente la coordonnée en x (entre 0 et 9) du pion
        :param y: Représente la coordonnée en y (entre 0 et 9) du pion
        :param color: Pion de couleur Blanc (0) ou Noir (1)
        """
        self.x = x
        self.y = y
        self.color = color
