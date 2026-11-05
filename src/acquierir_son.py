import sounddevice as sd
import soundfile as sf
import time




def enregistrer_micro(nom_fichier="data/musique_a_tester.wav", duree=15, sr=22050):

    for i in range(5,0,-1):
        print("l'enregistrement débute dans",i,"secondes")
        time.sleep(1)
    print("l'enregistrement débute pour une durée de",duree,'secondes')

    audio = sd.rec(int(duree * sr), samplerate=sr, channels=1) # Ici on utilise la fonction rec de sounddevice qui active le micro
    # elle prend en paramètre le nombre d'échantillons à réaliser, la fréquence d'échantillonnage et le nombre de canaux : mono/stereo

    sd.wait()  # On attend que l'enregistrement soit terminé
    sf.write(nom_fichier, audio, sr) # On écrit un fichier wav contenant les données acquises précedemment
    print(" Enregistrement terminé")