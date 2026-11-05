import os
import pickle
from lecture import *
from lecture_hashes import *

def construire_base():
    DOSSIER_MORCEAUX = "musiques-base-de-donnee"
    base_de_donnees = {}
    SR = 22050
    N_FFT = 2048
    HOP_LENGTH = 512

    for nom_fichier in os.listdir(DOSSIER_MORCEAUX):
        chemin = os.path.join(DOSSIER_MORCEAUX, nom_fichier)
        signal, sr = load_audio(chemin, sr=SR)
        S = compute_spectrogram(signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_amp = amplitude_spectrogramme(S)
        pics = find_peaks_in_spectrogram(S_amp, nb_peaks=5)
        Hashes = generate_hashes(pics, n_fft=N_FFT, sr=SR, hop_length=HOP_LENGTH, pics_a_coupler=2)
        identifiant = nom_fichier.replace(".wav", "")
        stock_donnees_dans_base(base_de_donnees, Hashes, identifiant)

    with open("base_de_donnees.pkl", "wb") as f:
        pickle.dump(base_de_donnees, f)

    print("Base de données créée.")

if __name__ == "__main__":
    construire_base()