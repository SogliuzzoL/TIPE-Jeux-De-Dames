from matplotlib import pyplot as plt

file_name = "C:\\Users\\lolo4\\PycharmProjects\\ia-jeu-de-dames\\score_1x1.csv"
sep = ";"

gen = []
moyenne_blancs = []
moyenne_noirs = []
mediane_blancs = []
mediane_noirs = []
ecart_type_blancs = []
ecart_type_noirs = []
max_blanc = []
max_noir = []

n_moyenne_glissante = 1

with open(file_name, 'r') as file:
    for line in file:
        line_sep = line.split(sep)
        gen.append(int(line_sep[0]))
        moyenne_blancs.append(float(line_sep[1]))
        moyenne_noirs.append(float(line_sep[2]))
        mediane_blancs.append(float(line_sep[3]))
        mediane_noirs.append(float(line_sep[4]))
        ecart_type_blancs.append(float(line_sep[5]))
        ecart_type_noirs.append(float(line_sep[6]))
        max_blanc.append(float(line_sep[7]))
        max_noir.append(float(line_sep[8].replace("\n", "")))


def lissage_courbe(l):
    moyenne_glissante = []
    for i in range(len(l) - n_moyenne_glissante):
        somme = 0
        for j in range(n_moyenne_glissante):
            somme += l[i + j]
        somme /= n_moyenne_glissante
        moyenne_glissante.append(somme)
    return moyenne_glissante


fig, ax = plt.subplots(4, 1)

# Moyenne Glissante
ax[0].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(moyenne_blancs), label="blancs")
ax[0].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(moyenne_noirs), label="noirs")
ax[0].set_title('Moyenne du score en fonction de la génération')

ax[1].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(mediane_blancs), label="blancs")
ax[1].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(mediane_noirs), label="noirs")
ax[1].set_title('Médiane du score en fonction de la génération')

ax[2].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(ecart_type_blancs), label="blancs")
ax[2].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(ecart_type_noirs), label="noirs")
ax[2].set_title('Ecart-type du score en fonction de la génération')

ax[3].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(max_blanc), label="blancs")
ax[3].plot(gen[:len(gen) - n_moyenne_glissante], lissage_courbe(max_noir), label="noirs")
ax[3].set_title('Score maximum en fonction de la génération')

plt.show()
