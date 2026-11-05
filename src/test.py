import os
import pickle  # pour sauvegarder la base
from lecture_hashes import *


SR = 22050
N_FFT = 2048
HOP_LENGTH = 512



fichier_test = "data"

chemin = os.path.join(fichier_test, "Ramenez-la-coupe-à-la-maison.wav")
signal, sr = load_audio(chemin, sr=SR)

S = compute_spectrogram(signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
S_amp = amplitude_spectrogramme(S)
pics = find_peaks_in_spectrogram(S_amp, nb_peaks=1)
empreintes = generate_hashes(pics, n_fft=N_FFT, sr=SR, hop_length=HOP_LENGTH, pics_a_coupler=2)

identifiant = "Ramenez-la-coupe-à-la-maison.wav".replace(".wav", "")


with open("base_de_donnees.pkl", "rb") as f:
    base_de_donnees = pickle.load(f)


piste,score = trouver_correspondance(empreintes,base_de_donnees)
print("le morceau que vous cherchez est surement : ",piste, "son score etait de :",score)