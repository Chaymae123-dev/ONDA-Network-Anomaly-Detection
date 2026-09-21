"""
preprocessing.py
----------------
Charge, nettoie, prépare et échantillonne les données CIC-IDS2017
pour l'entraînement du modèle.
"""

import numpy as np
import pandas as pd

from src.config import DOSSIER_DATA, COLONNE_LABEL, VALEUR_NORMALE, RANDOM_STATE


def charger_donnees():
    """Charge et fusionne les 8 fichiers .parquet du dossier data/."""
    fichiers = sorted(DOSSIER_DATA.glob("*.parquet"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier .parquet dans {DOSSIER_DATA}")

    print(f"[chargement] {len(fichiers)} fichier(s) trouvé(s) :")
    morceaux = []
    for f in fichiers:
        df = pd.read_parquet(f)
        print(f"   - {f.name} : {df.shape[0]} lignes")
        morceaux.append(df)

    data = pd.concat(morceaux, ignore_index=True)
    # Nettoyer les noms de colonnes (au cas où il resterait des espaces)
    data.columns = data.columns.str.strip()
    print(f"[chargement] Total fusionné : {data.shape[0]} lignes, {data.shape[1]} colonnes")
    return data


def nettoyer(df):
    """Supprime les valeurs infinies, manquantes et les doublons."""
    df = df.copy()

    # Remplacer les infinis par NaN (fréquent dans Flow Bytes/s, Flow Packets/s)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    avant = len(df)
    df.dropna(inplace=True)
    print(f"[nettoyage] {avant - len(df)} ligne(s) supprimée(s) (NaN / infini)")

    avant = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[nettoyage] {avant - len(df)} doublon(s) supprimé(s)")

    return df.reset_index(drop=True)


def creer_label_binaire(df):
    """Ajoute une colonne 'target' : 0 = normal (Benign), 1 = attaque."""
    df = df.copy()
    labels = df[COLONNE_LABEL].astype(str).str.strip().str.lower()
    df["target"] = (labels != VALEUR_NORMALE.lower()).astype(int)
    return df


def echantillonner(df, taille=300000):
    """
    Prend un échantillon aléatoire en gardant les proportions normal/attaque
    (échantillonnage stratifié). 'taille' = nombre de lignes voulu.
    """
    # Si on demande plus de lignes qu'il n'y en a, on garde tout
    if taille >= len(df):
        print(f"[échantillon] Dataset plus petit que {taille}, on garde tout ({len(df)}).")
        return df.reset_index(drop=True)

    fraction = taille / len(df)
    echantillon = (
        df.groupby("target", group_keys=False)
          .sample(frac=fraction, random_state=RANDOM_STATE)
          .reset_index(drop=True)
    )

    n_normal = (echantillon["target"] == 0).sum()
    n_attaque = (echantillon["target"] == 1).sum()
    print(f"[échantillon] {len(echantillon)} lignes retenues "
          f"(Normal : {n_normal} | Attaque : {n_attaque})")
    return echantillon


def preparer_donnees(taille_echantillon=300000):
    """Enchaîne : chargement -> nettoyage -> étiquette binaire -> échantillon."""
    df = charger_donnees()
    df = nettoyer(df)
    df = creer_label_binaire(df)
    df = echantillonner(df, taille=taille_echantillon)
    return df


if __name__ == "__main__":
    # Permet de tester ce fichier seul : python -m src.preprocessing
    data = preparer_donnees()
    print(data[["Label", "target"]].head(10))