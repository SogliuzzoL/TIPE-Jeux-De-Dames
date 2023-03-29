from plateau import *
import ia.deeplearning.ia as dl
import ia.minimax.ia as mm

if __name__ == "__main__":
    plateau = Plateau()

    positions = plateau.positions()
    print(positions)
    coups = coups_possibles(positions, 0)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
