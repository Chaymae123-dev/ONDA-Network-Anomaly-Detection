# ONDA Network Anomaly Detection

## 📌 Description

Projet de détection intelligente des anomalies du trafic réseau dans une infrastructure aéroportuaire, développé dans le cadre d'un projet de fin d'année à l'ONDA – Aéroport Nador El Aroui.

L'objectif est d'utiliser des techniques de Machine Learning pour analyser le trafic réseau, détecter les comportements anormaux et identifier différentes catégories d'attaques.

## 🎯 Objectifs

* Analyser et prétraiter des données de trafic réseau.
* Détecter les anomalies et les intrusions réseau.
* Entraîner un modèle de Machine Learning pour la classification des attaques.
* Évaluer les performances du modèle.
* Visualiser les résultats à travers une interface web interactive.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Streamlit
* Matplotlib
* Joblib

## 📊 Dataset

Le projet utilise le dataset public **CIC-IDS2017**, contenant différents scénarios de trafic réseau normal et malveillant.

Les fichiers du dataset ne sont pas inclus dans ce dépôt en raison de leur taille.

## 🤖 Modèle

Le modèle principal utilisé est **Random Forest** avec 100 arbres.

Résultats obtenus sur les données de test :

* Accuracy : **99.82 %**
* F1-score : **99.39 %**

## 📈 Résultats

Les résultats du projet sont disponibles dans le dossier `reports/`, notamment :

* Matrice de confusion
* Importance des variables
* Comparaison des modèles
* Résultats de l'évaluation

## 🖥️ Application

Une interface interactive développée avec **Streamlit** permet notamment de :

* Charger et analyser les données.
* Filtrer les résultats.
* Visualiser les prédictions.
* Consulter les résultats de détection.
* Exporter les résultats.

## 📁 Structure du projet

```text
ONDA-Network-Anomaly-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   └── Dataset CIC-IDS2017 non inclus
│
├── models/
│   └── modele_ids.joblib
│
├── reports/
│   ├── comparaison_modeles.csv
│   ├── comparaison_modeles.png
│   ├── importance_variables.png
│   └── matrice_confusion.png
│
├── src/
│   ├── compare_models.py
│   ├── config.py
│   ├── diagnostic.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Installation

Cloner le dépôt :

```bash
git clone https://github.com/Chaymae123-dev/ONDA-Network-Anomaly-Detection.git
cd ONDA-Network-Anomaly-Detection
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement sous Windows :

```bash
.venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## ▶️ Lancement de l'application

Pour lancer l'interface Streamlit :

```bash
streamlit run app/app.py
```

## 👩‍💻 Auteur

**Chaymae El Aissaoui**

Master Sciences et Techniques en Réseaux et Systèmes Informatiques
FST Settat
