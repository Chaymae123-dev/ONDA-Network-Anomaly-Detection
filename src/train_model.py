"""
train_model.py
--------------
Entraîne, évalue et sauvegarde le modèle de détection d'anomalies,
puis génère les graphiques (matrice de confusion, importance des variables).

Les métriques réelles (Accuracy, Precision, Recall, F1) sont enregistrées
dans le fichier du modèle, afin que l'interface affiche les vrais chiffres.

Lancement :  python -m src.train_model
"""

import joblib
import matplotlib
matplotlib.use("Agg")   # backend qui enregistre les images sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    CHEMIN_MODELE, RANDOM_STATE, TAILLE_TEST, COLONNE_LABEL, DOSSIER_REPORTS,
)
from src.preprocessing import preparer_donnees


def separer_x_y(df):
    """Sépare les variables explicatives (X) de la cible (y)."""
    colonnes_a_retirer = [COLONNE_LABEL, "target"]
    X = df.drop(columns=[c for c in colonnes_a_retirer if c in df.columns])
    # Sécurité : ne garder que les colonnes numériques
    X = X.select_dtypes(include=["number"])
    y = df["target"]
    return X, y, list(X.columns)


def tracer_matrice_confusion(y_test, y_pred):
    """Génère et enregistre la matrice de confusion dans reports/."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Attaque"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    chemin = DOSSIER_REPORTS / "matrice_confusion.png"
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    print(f"[graphique] Matrice de confusion enregistrée : {chemin}")


def tracer_importance_variables(modele, colonnes, top=15):
    """Trace les 'top' variables les plus importantes du modèle."""
    importances = modele.feature_importances_
    indices = np.argsort(importances)[-top:]   # les 'top' plus grandes

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(indices)), importances[indices], color="#0B2545")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([colonnes[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top} des variables les plus importantes")
    plt.tight_layout()
    chemin = DOSSIER_REPORTS / "importance_variables.png"
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    print(f"[graphique] Importance des variables enregistrée : {chemin}")


def entrainer():
    """Fonction principale d'entraînement."""
    # 1) Préparer les données
    df = preparer_donnees()

    # 2) Séparer X / y
    X, y, colonnes = separer_x_y(df)
    print(f"\n[train] {X.shape[1]} caractéristiques, {len(y)} exemples")

    # 3) Diviser train / test (stratify garde les proportions normal/attaque)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAILLE_TEST, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] Entraînement : {len(X_train)} | Test : {len(X_test)}")

    # 4) Mise à l'échelle
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5) Entraîner le Random Forest
    print("[train] Entraînement du Random Forest en cours...")
    modele = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,                 # utilise tous les cœurs du processeur
        class_weight="balanced",   # compense le déséquilibre normal/attaque
    )
    modele.fit(X_train_s, y_train)

    # 6) Évaluer
    y_pred = modele.predict(X_test_s)
    metriques = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print("\n===== RÉSULTATS =====")
    print(f"Accuracy  : {metriques['accuracy']:.4f}")
    print(f"Precision : {metriques['precision']:.4f}")
    print(f"Recall    : {metriques['recall']:.4f}")
    print(f"F1-Score  : {metriques['f1']:.4f}")
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Attaque"],
                                zero_division=0))

    # Générer les graphiques
    tracer_matrice_confusion(y_test, y_pred)
    tracer_importance_variables(modele, colonnes)

    # 7) Sauvegarder le modèle + scaler + colonnes + MÉTRIQUES RÉELLES
    paquet = {
        "modele": modele,
        "scaler": scaler,
        "colonnes": colonnes,
        "nom_modele": type(modele).__name__,
        "metriques": metriques,
    }
    joblib.dump(paquet, CHEMIN_MODELE)
    print(f"[train] Modèle sauvegardé : {CHEMIN_MODELE}")
    print(f"[train] Métriques enregistrées dans le modèle : "
          f"{ {k: round(v, 4) for k, v in metriques.items()} }")


if __name__ == "__main__":
    entrainer()