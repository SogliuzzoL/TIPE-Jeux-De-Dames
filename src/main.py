import datetime
import os.path
import random

from plateau import *
from ia.deeplearning.ia import *

if __name__ == "__main__":
    """
    Paramètre de la fenêtre
    """
    test_dames = True
    fast_simu = False
    human_vs_bot = True
    ia = True
    ia_training = True
    create_new_model = False
    model_start, model_end = None, None
    game_fps = 60
    case_depart = 0
    case_arrive = 0
    case_size = 72
    window_size = (1280, 720)
    window_name = "Jeu de dames"
    window_background_color = (236, 246, 249)
    total_win_noir = 0
    total_win_blanc = 0
    nb_parties = 0
    running = True
    waiting = False
    """
    Création d'un nouveau plateau
    """
    plateau = Plateau()
    if test_dames:
        plateau.pions = [Pion(30, 1, True), Pion(19, 0, False), Pion(8, 0, False), Pion(18, 0, False)]
    """
    Création IA
    """
    if ia:
        if create_new_model or not(os.path.isfile('model_start') and os.path.isfile('model_end')):
            model_start, model_end = start_training()
        else:
            model_start, model_end = load_model()
            if ia_training:
                start_training(model_start, model_end)

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
                            break
                    case_depart = 0
                    case_arrive = 0
        win = plateau.check_win()
        if win != -1:
            if win == 1:
                total_win_noir += 1
            elif win == 0:
                total_win_blanc += 1
            nb_parties += 1
            print(
                f'[{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}] - {str(float(total_win_blanc) / float(nb_parties) * 100)}% de parties gagnées par le blanc et {str(float(total_win_noir) / float(nb_parties) * 100)}% de parties gagnées par le noir (Blanc: {total_win_blanc}, Noir: {total_win_noir}, Total parties: {nb_parties})')
            plateau = Plateau()
        if fast_simu or human_vs_bot and plateau.round_side:
            coups = coups_possibles(plateau.positions(), plateau.round_side)
            if len(coups) == 0:
                plateau = Plateau()
            else:
                coup = coups[random.randint(0, len(coups) - 1)]
                if ia:
                    coup = run(plateau, model_start, model_end)
                plateau.jouer_coup(coup, plateau.round_side)
                if plateau.round_side:
                    plateau.round_side = 0
                else:
                    plateau.round_side = 1
                if not fast_simu:
                    print(f"Coup joué: {coup}")
                waiting = False
        """
        Informe le joueur des coups possibles
        """
        screen.fill(window_background_color)
        positions = plateau.positions()
        coups = coups_possibles(positions, plateau.round_side)
        real_coup = display_coup(coups)
        font = pygame.font.SysFont(pygame.font.get_fonts()[0], 24)
        text = font.render("Parties Gagnées :", True, (50, 50, 50))
        screen.blit(text, (5, 24 * 0 + 5))
        text = font.render(f"Blanc: {total_win_blanc}, Noirs: {total_win_noir}", True, (125, 125, 125))
        screen.blit(text, (5, 24 * 1 + 5))
        text = font.render(f"Nuls: {nb_parties - total_win_blanc - total_win_noir}, Total: {nb_parties}", True,
                           (125, 125, 125))
        screen.blit(text, (5, 24 * 2 + 5))
        text = font.render("Coup conseillé :", True, (50, 50, 50))
        screen.blit(text, (5, 24 * 3 + 5))
        ia_coups = display_coup([run(plateau, model_start, model_end)])
        if len(ia_coups) == 0:
            text = font.render(f'IA: {ia_coups[0]}', True, (125, 125, 125))
        screen.blit(text, (5, 24 * 4 + 5))
        text = font.render("Coups possibles :", True, (50, 50, 50))
        screen.blit(text, (5, 24 * 5 + 5))
        for i in range(1, len(real_coup) + 1):
            text = font.render(real_coup[i - 1], True, (125, 125, 125))
            screen.blit(text, (10, 24 * (i + 5) + 5))
        waiting = True
        """
        Modification graphique et mise à jour de la fenêtre
        """
        affichage_plateau(plateau, screen, case_depart)
        pygame.display.update()
        if not fast_simu:
            clock.tick(game_fps)
    pygame.quit()
