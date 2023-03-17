from plateau import *
import ia.deeplearning.ia as dl
import ia.minimax.ia as mm

if __name__ == "__main__":
    plateau = Plateau()

    positions = plateau.positions()
    positions = {22: [1, False], 6: [0, False], 9: [0, False], 10: [0, False], 17: [0, False], 18: [0, False], 20: [0, False], 27: [0, False], 28: [0, False], 29: [0, False], 36: [0, False], 39: [0, False], 50: [0, False]}
    coups = coups_possibles(positions, 1)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
