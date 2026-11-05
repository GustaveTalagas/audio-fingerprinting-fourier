import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from collections import Counter

#######################################
#   Telechargement du fichier audio   #
#######################################

def load_audio(filename, sr=22050):
    # Charge le fichier en mono
    # Parametre : filename chemin vers le ficier audio
    # parametre :  sr = (sample-rate) frequence d'échantillonage (Hz)
    # retourne signal : un tableau numpy des valeurs d'amplitude et sr la fréquence d'échantillonage

    signal, sr = librosa.load(filename, sr=sr, mono=True)
    signal = signal / np.max(np.abs(
        signal))  # sert à normaliser le signal. prend le maximum et divise toutes les valeurs par le maximum cela permet de simplifier la comparaison de signaux car il est plus simple de comparer des signaux dont l’échelle d’amplitude est similaire.

    return signal, sr


#######################################
#   Découpage en fenetres    #
#######################################

# Découpage du signal audio en courtes fenetres de temps. pour chaque fenetre on calcule une FFT. On assemblera ensuite ces FFT pour obtenir un spectrogramme.
# Cela nous permet d'observer quels patterns de fréquences apparaissent à tel instant.


def compute_spectrogram(signal, sr, n_fft=2048, hop_length=512):
    """
    Calcule le spectrogramme (STFT) d’un signal.

    :param signal: le signal audio (1D), amplitudes dans le domaine temporel
    :param sr: fréquence d’échantillonnage (Hz)
    :param n_fft: taille de la fenêtre de la FFT, c'est le nombre de fréquences dont on étudiera le poids.
    :param hop_length: nombre d’échantillons entre deux fenêtres
    :return: spectrogramme complexe en 2D
    """
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    return S


"""

— n_fft : taille de la fenêtre d’analyse (en échantillons). Si on est à 22050 Hz, 2048
correspond à ≈ 93 ms.
— hop_length : décalage entre deux fenêtres. Ici, 512 (soit 4 fois moins que 2048), ce
qui représente un overlap d’environ 75%.
— La sortie S est un tableau complexe 2D de dimension : chaque colonne du spectrogramme correspond à une fenetre. On a donc sur l'axe des abscisses les fenetres (temps),
 sur l'axe des ordonnées un nombre complexe contenant l'amplitude de chaque fréquence et la phase associée. Cela n'est pas vraiment exploitable on récupere donc le module ( l'amplitude )
pour tracer  le spectrogramme en amplitude.
"""


def amplitude_spectrogramme(S):
    return np.abs(S)


# visualisation de spectrogramme :


def show_spectrogram(S, sr, hop_length=512, y_axis='linear'):
    """
    Affiche le spectrogramme en dB pour une meilleure visualisation.

    :param S: spectrogramme complexe (2D) ou en amplitude (numpy array)
    :param sr: fréquence d’échantillonnage (int)
    :param hop_length: nombre d’échantillons entre deux fenêtres de la STFT
    :param y_axis: type d’échelle pour l’axe des fréquences ('linear', 'log', etc.)
    """
    # Conversion en décibels
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    # le ref=np.max sert a choisir la valeur en amplitude du 0dB qui sera donc la valeur maximale du tableau np.abs(S).

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis=y_axis, cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogramme")
    plt.tight_layout()
    plt.show()


#######################################
#   Extraction des points clés    #
#######################################

# Parmi toutes les fréquences dans une fenêtre, certaines se détachent nettement des autres, correspondant à des pics d’intensité.
# On va Repérer les points du spectrogramme où l’amplitude est localement maximale (points clés).
# De cette manière, au lieu de garder tout le spectrogramme, on ne stocke que quelques points par fenêtre ,réduisant ainsi la taille des données.

def find_peaks_in_spectrogram(S_amp, nb_peaks=5):
    """
    Trouve les pics de plus forte amplitude pour chaque colonne (frame).

    :param S_amp: spectrogramme en amplitude (matrice 2D : fréquences x temps)
    :param nb_peaks: nombre de pics à garder par frame
    :return: liste des indices des pics pour chaque frame (une liste par colonne)
    """
    n_frames = S_amp.shape[
        1]  # la fonction shape renvoie un tableau a 1D et deux colonnes, la première contient le nombre total de fréquence sur toutes les fenetres et la seconde le nombre total de fenetres
    peaks_by_frame = []  # Initialisation d'un tableau vide

    for frame_idx in range(n_frames):
        col = S_amp[:,
              frame_idx]  # On récupère la colonne correspondant à la frame actuelle : on récupère toutes les amplitudes des fréquences correspondant a la fenetre dans un tableau 1D

        peak_indices = np.argsort(col)[::-1][
                       :nb_peaks]  # La fonction argsort trie les indices qui trient un tableau par ordre croissant.  Ajouter [::-1] permet de tier les éléments par odre décroissant et [:nb_peaks] permet de ne garder que les np_peaks premiers indices, donc les indices des amplitudes les plus importantes.

        peaks_by_frame.append(peak_indices)  # On ajoute les indices des pics détectés à la liste

    return peaks_by_frame  # liste de listes : par exemple : [[12,142,43][76,380,757][1,543,9]] chaque sous liste contient les pics ayant lieu au cours d'une frame et l'indice de la sous liste nous renseigne sur : dans quelle frame a t on ces pics.



##########################################################
#   Création d'empreintes et d'une base de données   #
##########################################################
# Shazam ne garde pas toute la matrice peak by frame. Au lieu de cela on crée des desripteurs, des hashes pour chaque fenetre.
# Soit on prend uniquement les fréquences correspondant aux indices trouvés plus tot, soit on crée des paires de pics.
# Nous appliquerons la première méthode.

def generate_hashes(peaks_by_frame, n_fft=2048, sr=22050, hop_length=512, pics_a_coupler=2):

    hashes = [] #initialisation d'une liste vide qui va contenir tous les hashs générés
    time_per_frame = hop_length / sr # déja expliqué précédemment : temps entre deux frames du specrtogramme

    for t1, pics1 in enumerate(peaks_by_frame): # Boucle sur toutes les frames :  t_1 est l'indice de chaque frame et peaks1 est un tableau 1d d'indices poru chaque frame
        for idx1 in pics1: # Boucle sur chaque liste des indices des fréquences.
            freq1 = round(idx1 * sr / n_fft, 1) # On convertit les indices en fréquence comme expliqué dans le code lecture.py, on arrondit à 0,1
            for dt in range(1, pics_a_coupler + 1): # boucle sur les "pics à coupler" frames suivantes : on parcourt les pics a coupler frames suivantes
                t2 = t1 + dt # On prend le temps
                if t2 < len(peaks_by_frame): # On fait attention à ne pas dépasser la fin de l'enregistrement
                    for idx2 in peaks_by_frame[t2]: # on récupère les indices des fréquences des frames suivantes
                        freq2 = round(idx2 * sr / n_fft, 1) # On convertit les indices en fréquence
                        delta_t = round((t2 - t1) * time_per_frame, 2) # On calcule le temps entre la frame de reference et la frame étudiée actuellement = décalage temporel entre les deux pics.
                        hash = (freq1, freq2, delta_t) #On cree le fameux hash : un trouple contentant une fréquence dans une frame, une autre fréquence dans ue frame suivante et le temps entre ces deux fréquences.
                        ancrage_temporel = round(t1 * time_per_frame, 2) # ici on ancre temporellement la position du premier pic : temps écoulé entre le début de l'audio et la frame de référence qu'on étudie.
                        hashes.append((hash, ancrage_temporel)) #on remplit finalement le tableau
    return hashes # C'est un tableau a deux dimensions qui contient tous les hash


##########################################################
#   Stokage dans une base de données   #
##########################################################

def stock_donnees_dans_base(database, hashes, identifiant_piste):
    database[
        identifiant_piste] =  hashes # permet de dire a l'odinateur d'ajouter au dictionnaire "database" les hashes et de leur associer un identifiant.

    """
    par exemple la database sera : 

    database = {

    "morceau_1": fingerprints = [
    (440.0, 0.0),       # La note La (440 Hz) détectée à 0.0 s
    (523.25, 0.0232),   # Le Do (523 Hz) à 23 ms
    (659.25, 0.0464),   # Le Mi (659 Hz) à 46 ms
    (440.0, 0.0696),    # Encore le La à 69 ms
    (349.23, 0.0928),   # Le Fa à 92 ms
    (440.0, 0.116),     # Re-La
    (698.46, 0.1392),   # Le Sol# à 139 ms
    (440.0, 0.1624)     # Encore un La
],

        etc
}
    """


# base_de_donnees = {} # création d'une base de données.


##########################################################
#   Comparaison du sujet avec la base de donnee  #
##########################################################

#

def comparaison_hashs(hashes_inconnue, hashes_connue, delta_tol=0.05):
    """
    Compare deux listes de hashs et retourne un score.
    Le score correspond au nombre maximal de hashs alignés sur un même décalage temporel.
    """
    # Préparer une structure pour retrouver les temps associés à chaque hash dans le morceau connu
    hashs_base = {} # On crée un dictionnaire qui servira à contenir les temps d'encrage (tableau 1D contenant dans la première case le hash et dans la deuxième case un autre tableau contenant la liste des temps d'encrages associés. Il ne contient que des hashs différents)
    for hash_, t_base in hashes_connue: # On parcourt tous les hash du morceau connu
        if hash_ not in hashs_base: # Si le hash n'est pas déja dans la base
            hashs_base[hash_] = [] # Alors on crée une zone vide dans le dictionnaire asociée au hash en question
        hashs_base[hash_].append(t_base) # On ajoute ensuite le temps d'encrage dans la case correspondante. on peut donc avoir

    # On compte les votes pour chaque décalage temporel
    votes_par_décalage = Counter() # crée un compteur vide

    for hash_, t_test in hashes_inconnue: # Onq fait une boucle sur tous les hashs du morceau inconnu
        if hash_ in hashs_base: # Si le hash courant est dans le dictionnaire hash_base alors :
            for t_base in hashs_base[hash_]: # Pour chaque ancrage temporel correspondant au hash du dictionnaire :
                decalage = t_base - t_test # On calcule le décalage temporel
                decalage_arrondi = round(decalage / delta_tol) * delta_tol # Cela nous permet de regrouper les décalages proches dans un mee groupe
                votes_par_décalage[decalage_arrondi] += 1 # Cela permet de compter les occurences de chaque décalage et ce compteur est tenu indépendamment du hash courant, il compte les occurences des décalages par morceaux


    if votes_par_décalage: # permet d'éviter le cas limites ou l'extrait testé n'a pas de hash en commun avec le morceau de la base et donc le compteur est vide.
        return max(votes_par_décalage.values()) # on retourne le maximum de décalages egaux.
    else:
        return 0

def trouver_correspondance(hashes_inconnue, database): #Algo de comparaison simple
    meilleur_score = -1
    meilleur_piste = None
    for identifiant_piste, hashes_connue in database.items():
        sc = comparaison_hashs(hashes_inconnue, hashes_connue)
        if sc > meilleur_score:
            meilleur_score = sc
            meilleur_piste = identifiant_piste
    return meilleur_piste, meilleur_score

#################
# Ameliorations possibles
#################

# proposer plusieurs degrés de précision.
