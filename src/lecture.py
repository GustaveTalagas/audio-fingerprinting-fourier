import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


#######################################
#   Telechargement du fichier audio   #
#######################################

def load_audio( filename , sr =22050) :


    # Charge le fichier en mono
    # Parametre : filename chemin vers le ficier audio
    # parametre :  sr = (sample-rate) frequence d'échantillonage (Hz)
    # retourne signal : un tableau numpy des valeurs d'amplitude et sr la fréquence d'échantillonage

    signal,sr = librosa.load(filename , sr = sr , mono = True)
    signal = signal / np.max(np.abs(signal)) # sert à normaliser le signal. prend le maximum et divise toutes les valeurs par le maximum cela permet de simplifier la comparaison de signaux car il est plus simple de comparer des signaux dont l’échelle d’amplitude est similaire.

    return signal , sr

#######################################
    #   Découpage en fenetres    #
#######################################

#Découpage du signal audio en courtes fenetres de temps. pou chaque fenetre on calcule une FFT. On assemblera ensuite ces FFT pour obtenir un spectrogramme.
#Cela nous permet d'observer quels patterns de fréquences apparaissent à tel instant.


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


#visualisation de spectrogramme :




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

#Parmi toutes les fréquences dans une fenêtre, certaines se détachent nettement des autres, correspondant à des pics d’intensité.
# On va Repérer les points du spectrogramme où l’amplitude est localement maximale (points clés).
# De cette manière, au lieu de garder tout le spectrogramme, on ne stocke que quelques points par fenêtre ,réduisant ainsi la taille des données.



def find_peaks_in_spectrogram(S_amp, nb_peaks=5):
    """
    Trouve les pics de plus forte amplitude pour chaque colonne (frame).

    :param S_amp: spectrogramme en amplitude (matrice 2D : fréquences x temps)
    :param nb_peaks: nombre de pics à garder par frame
    :return: liste des indices des pics pour chaque frame (une liste par colonne)
    """
    n_frames = S_amp.shape[1]  #la fonction shape renvoie un tableau a 1D et deux colonnes, la première contient le nombre total de fréquence sur toutes les fenetres et la seconde le nombre total de fenetres
    peaks_by_frame = []      # Initialisation d'un tableau vide 

    for frame_idx in range(n_frames):

        col = S_amp[:, frame_idx]    # On récupère la colonne correspondant à la frame actuelle : on récupère toutes les amplitudes des fréquences correspondant a la fenetre dans un tableau 1D

        peak_indices = np.argsort(col)[::-1][:nb_peaks]     # La fonction argsort trie les indices qui trient un tableau par ordre croissant.  Ajouter [::-1] permet de tier les éléments par odre décroissant et [:nb_peaks] permet de ne garder que les np_peaks premiers indices, donc les indices des amplitudes les plus importantes.


        peaks_by_frame.append(peak_indices)            # On ajoute les indices des pics détectés à la liste

    return peaks_by_frame # liste de listes : par exemple : [[12,142,43][76,380,757][1,543,9]] chaque sous liste contient les pics ayant lieu au cours d'une frame et l'indice de la sous liste nous renseigne sur : dans quelle frame a t on ces pics.

##########################################################
  #   Création d'empreintes et d'une base de données   #
##########################################################
# Shazam ne garde pas toute la matrice peak by frame. Au lieu de cela on crée des desripteurs, des hashes pour chaque fenetre.
# Soit on prend uniquement les fréquences correspondant aux indices trouvés plus tot, soit on crée des paires de pics.
# Nous appliquerons la première méthode.



def generate_fingerprints(peaks_by_frame, n_fft=2048, sr=22050, hop_length=512):
    """
    Génère un ensemble de "fingerprints" pour un morceau audio à partir des pics fréquentiels par frame.

    :param peaks_by_frame: liste où peaks_by_frame[t] = liste des indices de pics pour le frame t
    :param n_fft: taille de la FFT (résolution fréquentielle)
    :param sr: fréquence d’échantillonnage (Hz)
    :param hop_length: nombre d’échantillons entre deux frames
    :return: une liste de tuples (freq_en_Hz, temps_en_secondes)
    """
    fingerprints = []

    time_per_frame = float(hop_length) / sr    # on calcule le temps entre deux analyses du signal : on crée une nouvelle fenêtre tous les 512 échantillons,on échantillonne à un taux de 22050 Hz donc le temps entre deux fenêtres correspond a 512/22050 =  float(hop_length) / sr

    for t, peak_indices in enumerate(peaks_by_frame): #la commande enumerate permet de séparer les tableaux 2d : d'un coté t absorbe les indices des sous tableaux(le nombre de frames écoulées), de l'autre coté peak_indices absorbe les sous tableaux.
        time = t * time_per_frame  #calcul du temps total écoulé, t est le nombre de frames parcourues et time_per_frame est le temps écoulé entre chaque fenêtre ce qui nous donne donc le temps total écoulé. ainsi, si t = 4,
        # peak indices correspond aux indices (fréquences) des amplitudes. s
        for idx in peak_indices: # parcourt chaque sous tableau
            freq_hz = idx * sr / float(n_fft)  #Rappel : les sous tableau contiennent les indices des pics d'intensité de chaque frame. Chaque indice correspond à une fréquence. On récupère ici la fréquence correspondant à l'indice : on étudie une plage de fréquence de 22 000 Hz qu'on découpe en 2048 frames. chaque morceau
            fingerprints.append((freq_hz, time)) # création d'un tableau 2D qui contient les fréquences et le temps ou elles ont été mesurées.

    return fingerprints


##########################################################
        #   Stokage dans une base de données   #
##########################################################

def stock_donnees_dans_base(database,fingerprints,identifiant_piste ):


    database[identifiant_piste] = fingerprints # permet de dire a l'odinateur d'ajouter au dictionnaire "database" les fingerprints et de leur associer un identifiant.

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
def comparaison_fingerprints(fingerprint_inconnue, fingerprint_connue, freq_tolerance = 2, tolerence_temporelle = 0.1):
    score  = 0
    for (f_i, t_i) in fingerprint_inconnue :
        for (f_c, t_c) in fingerprint_connue:
            if abs(f_i -f_c) <freq_tolerance and (t_i -t_c) <tolerence_temporelle :
                score +=1
    return score

def trouver_correspondance(fingerprint_inconnue, database):
    meilleur_score = -1
    meilleur_piste = None
    for identifiant_piste, fingerprint_connue in database.items() :
        sc = comparaison_fingerprints(fingerprint_inconnue, fingerprint_connue)
        if sc > meilleur_score :
            meilleur_score = sc
            meilleur_piste = identifiant_piste
    #if meilleur_score < 50000 :
        #return "Aucun morceau de la base de donnée ne correspond suffisamment"
    return meilleur_piste, meilleur_score






#################
# Ameliorations possibles
#################

#proposer plusieurs degrés de précision.