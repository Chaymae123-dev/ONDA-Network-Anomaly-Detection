"""
config.py — centralise les chemins et paramètres du projet.
"""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE / "data"
DOSSIER_MODELS = RACINE / "models"
DOSSIER_REPORTS = RACINE / "reports"

for d in (DOSSIER_DATA, DOSSIER_MODELS, DOSSIER_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

CHEMIN_MODELE = DOSSIER_MODELS / "modele_ids.joblib"

COLONNE_LABEL = "Label"
VALEUR_NORMALE = "Benign"

RANDOM_STATE = 42
TAILLE_TEST = 0.2