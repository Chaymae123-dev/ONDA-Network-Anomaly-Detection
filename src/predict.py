"""
predict.py
----------
Charge le modèle et prédit, pour chaque flux, Normal (0) ou Attaque (1).
Renvoie aussi la probabilité (confiance) de chaque prédiction.
"""
import numpy as np
import pandas as pd
import joblib
from src.config import CHEMIN_MODELE


def charger_modele():
    if not CHEMIN_MODELE.exists():
        raise FileNotFoundError("Modèle introuvable : lance python -m src.train_model")
    return joblib.load(CHEMIN_MODELE)


def preparer_pour_prediction(df, colonnes_attendues):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    for col in colonnes_attendues:
        if col not in df.columns:
            df[col] = 0
    return df[colonnes_attendues]


def predire(df):
    """Renvoie (df_resultat, predictions, confiances)."""
    paquet = charger_modele()
    modele, scaler, colonnes = paquet["modele"], paquet["scaler"], paquet["colonnes"]
    X = preparer_pour_prediction(df, colonnes)
    X_scaled = scaler.transform(X)
    predictions = modele.predict(X_scaled)
    # Confiance = probabilité de la classe prédite
    try:
        proba = modele.predict_proba(X_scaled)
        confiances = proba.max(axis=1)
    except Exception:
        confiances = np.ones(len(predictions))
    resultat = df.copy()
    resultat["Prédiction"] = np.where(predictions == 1, "Attaque", "Normal")
    return resultat, predictions, confiances