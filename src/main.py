from plateau import *
import ia.deeplearning.ia as dl
import ia.minimax.ia as mm

if __name__ == "__main__":
    plateau = Plateau()

    positions = plateau.positions()

    #positions = {41: [0, True], 42: [0, True], 43: [0, True], 44: [0, True], 45: [0, True]}
    positions = {46: [0, True], 47: [0, True], 48: [0, True], 49: [0, True], 50: [0, True]}
    print(positions)
    coups = coups_possibles(positions, 0)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
