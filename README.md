# audio-fingerprinting-fourier

Projet consistant en la réalisation d'une vidéo sur le thème: *Période et fréquence*. Nous avons choisi d'aborder cette thématique sous l'angle de la transformée de fourier et la reconnaissance musicale. 

## Auteurs & Encadrement
* **Équipe projet :** G. Talagas, A. Bayraktar, T. Bracco
* **Encadrant :** Dr. Pierre Pansu (Université Paris-Saclay)

## Démarche et Fonctionnement
Ce projet illustre l'application de la **Transformée de Fourier à court terme (STFT)** au traitement du signal audio. Le programme :
1. Analyse un extrait sonore (capture micro ou fichier) via la bibliothèque `librosa`.
2. Génère un spectrogramme en extrayant les pics d'intensité maximale.
3. Crée des empreintes acoustiques uniques par un algorithme de **hachage temps-fréquence**.
4. Identifie le morceau en évaluant la cohérence des décalages temporels avec une base de données de référence.
### Animation Mathématique (Manim)
Pour expliquer visuellement la transformée de fourier et notre code nous avons programmé une animation avec **Manim**, sur python.
* **[Visionner l'animation explicative](docs/Extrait-Manim.mp4)**

## Architecture du Code (Dossier `/src`)
* `lecture_hashes.py` : Cœur mathématique et algorithmique (calcul STFT, extraction des pics, création des empreintes et logique d'alignement temporel).
* `base_de_donnees.py` : Script d'initialisation analysant les fichiers audio complets pour constituer la base de données de référence (`.pkl`).
* `acquierir_son.py` : Module gérant l'activation du microphone et l'enregistrement de l'extrait à tester.
* `retrouver-piste.py` : Script principal d'exécution liant la capture, l'analyse et la comparaison pour retourner le titre reconnu.
* `quantifier-performance.py` : Algorithme de benchmark automatisé évaluant la fiabilité du modèle sur des extraits tronqués aléatoirement.

## Rapport d'Étude
L'explication mathématique détaillée de la transformation de Fourier (du modèle continu à l'application discrète) et l'analyse de nos résultats statistiques sont disponibles dans le document de synthèse :
*  **[Consulter le rapport de projet (PDF)](docs/Rapport-Fourier.pdf)**
