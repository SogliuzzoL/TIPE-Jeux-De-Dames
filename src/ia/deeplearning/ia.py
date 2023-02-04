def make_prediction(plateau):
    inputs = [[0, 0, 0] for i in range(51)]
    for pion in plateau.pions:
        inputs[pion.emplacement] = [1, pion.color, int(pion.dame)]
    real_inputs = []
    for datas in inputs:
        for data in datas:
            real_inputs.append(data)
    print(real_inputs)
