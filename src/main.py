import pygame

from plateau import *
import ia.deeplearning.ia as dl
import ia.minimax.ia as mm

if __name__ == "__main__":
    """
    Paramètre de la fenêtre
    """
    window_size = (1280, 720)
    window_name = "Jeu de dames"
    window_background_color = (236, 246, 249)
    running = True
    """
    Création d'un nouveau plateau
    """
    plateau = Plateau()

    # positions = plateau.positions()

    # positions = {28: [0, True], 43: [0, True]}
    # positions = {41: [0, True], 42: [0, True], 43: [0, True], 44: [0, True], 45: [0, True]}
    # positions = {46: [0, True], 47: [0, True], 48: [0, True], 49: [0, True], 50: [0, True]}
    # print(positions)

    # coups = coups_possibles(positions, 0)
    # print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")

    """
    Création de la fenêtre
    """
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption(window_name)
    clock = pygame.time.Clock()
    """
    Boucle de la fenêtre
    """
    while running:
        """
        Gestion des évènements de la fenêtre
        """
        for event in pygame.event.get():
            """
            Utilisateur quitte la fenêtre
            """
            if event.type == pygame.QUIT:
                running = False
        """
        Modification graphique et mise à jour de la fenêtre
        """
        screen.fill(window_background_color)
        affichage_plateau(plateau, screen)
        pygame.display.update()
        clock.tick(60)
    pygame.quit()
