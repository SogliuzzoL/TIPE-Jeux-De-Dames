from plateau import *
import ia.deeplearning.ia as dl
import ia.minimax.ia as mm


if __name__ == "__main__":
    plateau = Plateau()

    coups = coups_possibles(plateau.positions(), 1)
    print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
    model = dl.NeuralNetwork()
    dl.make_prediction(plateau, model)
