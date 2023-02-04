from plateau import *
import src.ia.deeplearning.ai as dl
import src.ia.minimax.ai as mm


if __name__ == "__main__":
    plateau = Plateau()

    coups = coups_possibles(plateau.positions(), 1)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
