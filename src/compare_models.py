"""
compare_models.py
-----------------
Compare trois algorithmes de Machine Learning sur le même jeu de données :
    - Decision Tree  (arbre de décision)
    - Random Forest  (forêt aléatoire)
    - SVM            (machine à vecteurs de support)

Pour chacun on mesure : Accuracy, Precision, Recall, F1-Score et le temps
d'entraînement. Un tableau et un graphique sont générés dans reports/.

⚠️ Le SVM est BEAUCOUP plus lent. Il est entraîné sur un sous-échantillon
réduit (--taille-svm). À mentionner honnêtement dans le rapport.

Lancement :
    python -m src.compare_models
    python -m src.compare_models --taille 100000 --taille-svm 10000
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, TAILLE_TEST, DOSSIER_REPORTS
from src.preprocessing import preparer_donnees
from src.train_model import separer_x_y


def evaluer(nom, modele, X_train, X_test, y_train, y_test):
    """Entraîne un modèle, mesure le temps et calcule les 4 métriques."""
    print(f"\n[{nom}] Entraînement en cours...")
    debut = time.time()
    modele.fit(X_train, y_train)
    duree = time.time() - debut

    y_pred = modele.predict(X_test)
    resultat = {
        "Algorithme": nom,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Temps (s)": duree,
        "Exemples": len(X_train),
    }
    print(f"[{nom}] Terminé en {duree:.1f}s — "
          f"Accuracy {resultat['Accuracy']:.4f} | F1 {resultat['F1-Score']:.4f}")
    return resultat


def tracer_comparaison(df_resultats):
    """Graphique en barres groupées comparant les 4 métriques."""
    metriques = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = range(len(df_resultats))
    largeur = 0.2
    couleurs = ["#0B2545", "#0E5C43", "#D9A521", "#C1272D"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, c) in enumerate(zip(metriques, couleurs)):
        positions = [p + i * largeur for p in x]
        ax.bar(positions, df_resultats[m], largeur, label=m, color=c)

    ax.set_xticks([p + 1.5 * largeur for p in x])
    ax.set_xticklabels(df_resultats["Algorithme"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des algorithmes de détection")
    ax.legend(loc="lower right")
    plt.tight_layout()

    chemin = DOSSIER_REPORTS / "comparaison_modeles.png"
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    print(f"\n[graphique] Comparaison enregistrée : {chemin}")


def comparer(taille=300000, taille_svm=20000):
    """Lance la comparaison des trois algorithmes."""
    df = preparer_donnees(taille_echantillon=taille)
    X, y, colonnes = separer_x_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAILLE_TEST, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\n[comparaison] Entraînement : {len(X_train_s)} | Test : {len(X_test_s)}")

    resultats = []

    resultats.append(evaluer(
        "Decision Tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        X_train_s, X_test_s, y_train, y_test,
    ))

    resultats.append(evaluer(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                               n_jobs=-1, class_weight="balanced"),
        X_train_s, X_test_s, y_train, y_test,
    ))

    n_svm = min(taille_svm, len(X_train_s))
    print(f"\n[SVM] Sous-échantillon de {n_svm} exemples (algorithme lent).")
    resultats.append(evaluer(
        "SVM",
        SVC(kernel="rbf", random_state=RANDOM_STATE, class_weight="balanced"),
        X_train_s[:n_svm], X_test_s, y_train[:n_svm], y_test,
    ))

    df_res = pd.DataFrame(resultats)

    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF")
    print("=" * 70)
    affichage = df_res.copy()
    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        affichage[m] = (affichage[m] * 100).round(2).astype(str) + " %"
    affichage["Temps (s)"] = affichage["Temps (s)"].round(1)
    print(affichage.to_string(index=False))
    print("=" * 70)

    meilleur = df_res.loc[df_res["F1-Score"].idxmax(), "Algorithme"]
    print(f"\n➜ Meilleur F1-Score : {meilleur}")

    chemin_csv = DOSSIER_REPORTS / "comparaison_modeles.csv"
    df_res.to_csv(chemin_csv, index=False)
    print(f"[fichier] Tableau enregistré : {chemin_csv}")

    tracer_comparaison(df_res)
    return df_res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Comparaison des algorithmes")
    p.add_argument("--taille", type=int, default=300000,
                   help="Nombre de lignes du jeu de données (défaut 300000)")
    p.add_argument("--taille-svm", type=int, default=20000,
                   help="Nombre d'exemples pour le SVM (défaut 20000)")
    args = p.parse_args()
    comparer(taille=args.taille, taille_svm=args.taille_svm)