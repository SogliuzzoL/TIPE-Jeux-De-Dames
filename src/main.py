import random

from plateau import *

if __name__ == "__main__":
    """
    Paramètre de la fenêtre
    """
    test_dames = False
    fast_simu = False
    human_vs_bot = True
    game_fps = 60
    case_depart = 0
    case_arrive = 0
    case_size = 72
    window_size = (1280, 720)
    window_name = "Jeu de dames"
    window_background_color = (236, 246, 249)
    running = True
    waiting = False
    """
    Création d'un nouveau plateau
    """
    plateau = Plateau()
    if test_dames:
        plateau.pions = [Pion(28, 0, True), Pion(48, 1, True)]
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
            Utilisateur clique sur la fenêtre
            """
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_x -= (screen.get_size()[0] - screen.get_size()[1]) // 2
                case_x = mouse_x // case_size
                case_y = mouse_y // case_size
                case_number = 0
                if not case_y % 2 and case_x % 2 or case_y % 2 and not case_x % 2:
                    case_number = case_x // 2 + 10 * case_y // 2 + 1
                if case_depart == 0:
                    case_depart = case_number
                else:
                    case_arrive = case_number
                    coups = coups_possibles(plateau.positions(), plateau.round_side)
                    for coup in coups:
                        if coup.startswith(str(case_depart)) and coup.endswith(str(case_arrive)):
                            plateau.jouer_coup(coup, plateau.round_side)
                            if plateau.round_side:
                                plateau.round_side = 0
                            else:
                                plateau.round_side = 1
                            print(f"Coup joué: {coup}")
                            waiting = False
                    case_depart = 0
                    case_arrive = 0
        win = plateau.check_win()
        if (win == 0 or win == 1) and not test_dames:
            plateau = Plateau()
        if fast_simu or human_vs_bot and plateau.round_side:
            coups = coups_possibles(plateau.positions(), plateau.round_side)
            if len(coups) == 0:
                plateau = Plateau()
            else:
                coup = coups[random.randint(0, len(coups) - 1)]
                plateau.jouer_coup(coup, plateau.round_side)
                if plateau.round_side:
                    plateau.round_side = 0
                else:
                    plateau.round_side = 1
                print(f"Coup joué: {coup}")
                waiting = False
        """
        Informe le joueur des coups possibles
        """
        screen.fill(window_background_color)
        positions = plateau.positions()
        coups = coups_possibles(positions, plateau.round_side)
        font = pygame.font.SysFont(pygame.font.get_fonts()[0], 24)
        text = font.render("Coups possibles :", True, (50, 50, 50))
        screen.blit(text, (5, 5))
        for i in range(1, len(coups) + 1):
            text = font.render(coups[i - 1], True, (125, 125, 125))
            screen.blit(text, (10, 24 * i + 5))
        waiting = True
        """
        Modification graphique et mise à jour de la fenêtre
        """
        affichage_plateau(plateau, screen)
        pygame.display.update()
        if not fast_simu:
            clock.tick(game_fps)
    pygame.quit()
