# code pour quantifier la performance du logiciel.
# l'idée est de découper tous les morceaux de la base en morceaux plus courts de 2 sec de manière aléatoire.
# On réalise alors une analyse par fourier et si le programme retrouve le bon morceau on ajoute 1 au score par morceau.
#L'idée est de tester pour beaucoup d'échantillons pour faire des statistiques.
#pour cela on peut créer une base de donnée fractionnée qui contient donc des milliers d'extraits

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from lecture_hashes import *







def charger_tous_les_signaux(dossier, sr):
    fichiers_audio = {}
    for nom_fichier in os.listdir(dossier):
        if nom_fichier.endswith(".wav"):
            chemin = os.path.join(dossier, nom_fichier)
            signal, _ = load_audio(chemin, sr=sr)
            identifiant = nom_fichier.replace(".wav", "")
            fichiers_audio[identifiant] = signal
    return fichiers_audio




######################################################


def test_duree(duree_extrait, fichiers_audio, base_de_donnees, sr=22050, nb_tests_par_morceau=1):
    n_fft = 2048
    hop_length = 512
    nb_total = 0
    nb_reussis = 0

    for identifiant_attendu, signal in fichiers_audio.items():
        nb_echantillons_total = len(signal)
        nb_echantillons_par_extrait = int(duree_extrait * sr)

        if nb_echantillons_total < nb_echantillons_par_extrait:
            continue

        for _ in range(nb_tests_par_morceau):
            debut = np.random.randint(0, nb_echantillons_total - nb_echantillons_par_extrait)
            extrait = signal[debut:debut + nb_echantillons_par_extrait]

            S = compute_spectrogram(extrait, sr=sr, n_fft=n_fft, hop_length=hop_length)
            S_amp = amplitude_spectrogramme(S)
            pics = find_peaks_in_spectrogram(S_amp, nb_peaks=5)
            hashes = generate_hashes(pics, n_fft=n_fft, sr=sr, hop_length=hop_length, pics_a_coupler=2)

            identifiant_trouve, _ = trouver_correspondance(hashes, base_de_donnees)

            if identifiant_trouve == identifiant_attendu:
                nb_reussis += 1
            nb_total += 1

    pourcentage = round((nb_reussis / nb_total) * 100, 2) if nb_total > 0 else 0
    return duree_extrait, pourcentage

# === Fonction principale graphique avec parallélisation ===
def graph_parallel():
    SR = 22050
    DOSSIER = "musiques-base-de-donnee"
    NB_TESTS_PAR_MORCEAU = 10

    print("Chargement des signaux audio...")
    fichiers_audio = charger_tous_les_signaux(DOSSIER, sr=SR)

    print("Chargement de la base de données...")
    with open("base_de_donnees.pkl", "rb") as f:
        base = pickle.load(f)

    intervalle_temporel = np.linspace(0.05, 0.5, 10)

    print("Lancement des tests en parallèle...")
    pourcentages_temps = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(test_duree, duree, fichiers_audio, base, SR, NB_TESTS_PAR_MORCEAU)
            for duree in intervalle_temporel
        ]
        for future in as_completed(futures):
            duree, taux = future.result()
            pourcentages_temps.append((duree, taux))
            print(f"Durée {round(duree, 2)} sec → {taux}%")

    pourcentages_temps.sort()
    x, y = zip(*pourcentages_temps)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', label='Taux de reconnaissance')
    plt.xlabel("Durée de l'extrait (s)")
    plt.ylabel("Taux de reconnaissance (%)")
    plt.title("Performance du système en fonction de la durée de l'extrait")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graph_parallel()