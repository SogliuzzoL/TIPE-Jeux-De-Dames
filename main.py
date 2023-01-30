from plateau import *

plateau = Plateau()

coups = coups_possibles(plateau.positions(), 1)
print(f"Nombres de coups: {len(coups)}. Coups possibles:{coups}")
