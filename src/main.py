from plateau import *
import ia.deeplearning.ai as dl
import ia.minimax.ai as mm


if __name__ == "__main__":
    plateau = Plateau()

    coups = coups_possibles(plateau.positions(), 1)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
    dl.make_prediction(plateau)
