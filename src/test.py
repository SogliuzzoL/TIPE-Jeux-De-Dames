import pygame

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre du jeu
largeur_fenetre = 600
hauteur_fenetre = 600

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ROUGE = (255, 0, 0)

# Dimensions de la grille
taille_case = largeur_fenetre // 8

# Création de la fenêtre du jeu
fenetre = pygame.display.set_mode((largeur_fenetre, hauteur_fenetre))
pygame.display.set_caption("Jeu de Dames")

# Boucle principale du jeu
def jeu_dames():
    plateau = init_plateau()
    joueur_actif = 1  # 1 pour les pions blancs, 2 pour les pions noirs

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Logique du jeu ici

        dessiner_plateau(plateau)

        pygame.display.update()

# Fonction pour initialiser le plateau de jeu
def init_plateau():
    plateau = [[0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 != 0:
                if i < 3:
                    plateau[i][j] = 1  # Pion blanc
                elif i > 4:
                    plateau[i][j] = 2  # Pion noir
    return plateau

# Fonction pour dessiner le plateau de jeu
def dessiner_plateau(plateau):
    fenetre.fill(BLANC)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                pygame.draw.rect(fenetre, NOIR, (i * taille_case, j * taille_case, taille_case, taille_case))
            else:
                pygame.draw.rect(fenetre, ROUGE, (i * taille_case, j * taille_case, taille_case, taille_case))

            if plateau[i][j] == 1:
                pygame.draw.circle(fenetre, BLANC, (i * taille_case + taille_case // 2, j * taille_case + taille_case // 2), taille_case // 2 - 5)
            elif plateau[i][j] == 2:
                pygame.draw.circle(fenetre, NOIR, (i * taille_case + taille_case // 2, j * taille_case + taille_case // 2), taille_case // 2 - 5)

jeu_dames()
