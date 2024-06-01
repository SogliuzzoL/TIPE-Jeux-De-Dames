import os

from matplotlib import pyplot as plt

file_dir = "\\\\DESKTOP-VR8PC8A\\Users\\Proxmox\\Documents\\ia-jeu-de-dames\\src\\"
sep = ";"
start = "score_2x"
end = ".csv"

surface = False

n_moyenne_glissante = 10


def lissage_courbe(l):
    moyenne_glissante = []
    for i in range(len(l)):
        somme = 0
        for j in range(min(i, n_moyenne_glissante)):
            somme += l[i - j]
        somme /= n_moyenne_glissante
        moyenne_glissante.append(somme)
    return moyenne_glissante


X = []
Y = []
Z = []

for file in os.listdir(file_dir):
    gen = []
    moyenne_blancs = []
    moyenne_noirs = []
    mediane_blancs = []
    mediane_noirs = []
    ecart_type_blancs = []
    ecart_type_noirs = []
    max_blanc = []
    max_noir = []
    if file[:len(start)] == start and file[-len(end):] == end:
        with open(file_dir + file, 'r') as file:
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
        noirs = lissage_courbe(moyenne_noirs)
        blancs = lissage_courbe(moyenne_blancs)
        moyenne = []
        for i in range(len(noirs)):
            moyenne.append((noirs[i] + blancs[i]) / 2)
        if not surface:
            plt.plot(gen, moyenne, label=file.name.split('\\')[-1].replace('.csv', '').replace('score_', ''))
        X.append(int(file.name.split('\\')[-1][len(start)]))
        Y.append(int(file.name.split('x')[-1].split('.')[0]))
        Z.append((lissage_courbe(moyenne_noirs)[-1] + lissage_courbe(moyenne_blancs)[-1]) / 2)

meilleur_indice = Z.index(max(Z))
print(f'Meilleur modèle: Longueur = {X[meilleur_indice]}, Largeur={Y[meilleur_indice]}')

if surface:
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='magma')
    ax.set_xlabel('Largeur du modèle')
    ax.set_ylabel('Longueur du modèle')
    ax.set_zlabel('Score')

plt.legend()
plt.show()
