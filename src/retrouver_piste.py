from acquierir_son import *
from base_de_donnees import *
from lecture_hashes import * # On importe toutes les fonctions d'analyse du signal

def retrouver_piste(SR = 22050,N_FFT = 2048,HOP_LENGTH = 512):

    enregistrer_micro(nom_fichier="data/musique_a_tester.wav",duree=4, sr=22050)

    fichier_test = "data/musique_a_tester.wav"

    chemin = fichier_test
    signal, sr = load_audio(chemin, sr=SR)

    S = compute_spectrogram(signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_amp = amplitude_spectrogramme(S)
    pics = find_peaks_in_spectrogram(S_amp, nb_peaks=5)
    empreintes = generate_hashes(pics, n_fft=N_FFT, sr=SR, hop_length=HOP_LENGTH,pics_a_coupler=2)

    #identifiant = "musique à tester.wav".replace(".wav", "")


    with open("base_de_donnees.pkl", "rb") as f:
        base_de_donnees = pickle.load(f)


    show_spectrogram(S,SR,HOP_LENGTH,y_axis='linear')


    piste,score = trouver_correspondance(empreintes,base_de_donnees)
    print("le morceau que vous cherchez est surement : ",piste, "son score etait de :",score)

    return None





retrouver_piste(SR = 22050,N_FFT = 2048,HOP_LENGTH = 512)