# Chest X-ray Classification

Ce projet vise à classifier les radiographies thoraciques (Chest X-ray) en différentes catégories à l'aide de plusieurs modèles de machine learning et deep learning.

## Structure du projet

| Dossier/Fichier         | Description                                      |
|------------------------|--------------------------------------------------|
| `config.py`            | Fichier de configuration du projet               |
| `main.ipynb`           | Notebook principal pour l'exécution du projet    |
| `src/models/`          | Définitions des modèles (CNN, SVM, etc.)        |
| `src/datasets/`        | Chargement et gestion des jeux de données        |
| `src/utils/`           | Fonctions utilitaires (affichage, etc.)         |
| `requirements.txt`     | Dépendances Python requises                     |
| `datasets_urls.txt`    | URLs pour télécharger les jeux de données        |
| `models_cache/`        | Modèles entraînés sauvegardés                   |

## Installation

1. Installez Python 3.12.9
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Téléchargez les jeux de données en utilisant les liens dans `datasets_urls.txt`.

## Utilisation

- Lancez le notebook `main.ipynb` pour entraîner, évaluer et comparer les modèles.
- Les modèles disponibles incluent :
  - CNN (TensorFlow)
  - Forêt aléatoire (Random Forest)
  - KNN
  - SVM

## Visualisation

Le notebook propose des visualisations des performances (accuracy, precision, recall, f1-score) et des matrices de confusion pour chaque modèle.

## Auteurs
- Projet académique, 2025
