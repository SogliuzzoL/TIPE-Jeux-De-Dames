from plateau import *
import ia.deeplearning.ia as dl
import ia.minimax.ia as mm

if __name__ == "__main__":
    plateau = Plateau()

    positions = plateau.positions()
    positions = {17: [1, False], 34: [1, False], 11: [0, False], 12: [0, False], 21: [0, False], 22: [0, False], 29: [0, False], 30: [0, False], 39: [0, False], 40: [0, False], 20: [0, False]}
    coups = coups_possibles(positions, 1)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
