"""
diagnostic.py
-------------
Analyse les flux détectés comme anomalies et propose un DIAGNOSTIC :
type d'attaque probable, niveau de risque, causes probables et recommandations.

Le diagnostic par TYPE utilise la vraie étiquette (colonne Label) si elle est
présente dans le fichier — il varie donc selon le fichier analysé. Sinon, il
retombe sur des règles d'expert appliquées aux caractéristiques du flux.
"""

import numpy as np
import pandas as pd

CATALOGUE = {
    "PortScan": {
        "risque": "Élevé",
        "causes": [
            "Grand nombre de connexions vers des ports différents.",
            "Durée des flux très courte.",
            "Comportement typique d'un balayage de ports.",
        ],
        "recommandations": [
            "Bloquer l'adresse source au niveau du pare-feu.",
            "Fermer les ports inutilisés et limiter les services exposés.",
            "Activer une limitation du taux de connexions (rate limiting).",
        ],
    },
    "DDoS / DoS": {
        "risque": "Critique",
        "causes": [
            "Débit de paquets anormalement élevé.",
            "Volume de paquets sortants très important.",
            "Saturation possible du service ciblé.",
        ],
        "recommandations": [
            "Activer la protection anti-DDoS et le filtrage en amont.",
            "Répartir la charge (load balancing) sur plusieurs serveurs.",
            "Contacter le fournisseur d'accès pour un filtrage réseau.",
        ],
    },
    "Brute Force (FTP/SSH)": {
        "risque": "Élevé",
        "causes": [
            "Connexions répétées vers les ports d'administration (21, 22).",
            "Flux courts et nombreux vers le même service.",
            "Tentatives d'authentification successives probables.",
        ],
        "recommandations": [
            "Limiter le nombre de tentatives de connexion.",
            "Imposer une authentification par clé ou à deux facteurs.",
            "Restreindre l'accès SSH/FTP à des adresses autorisées.",
        ],
    },
    "Attaque Web": {
        "risque": "Moyen",
        "causes": [
            "Trafic anormal vers les ports web (80, 443, 8080).",
            "Taille de paquets inhabituelle pour des requêtes HTTP.",
            "Requêtes potentiellement malveillantes (injection, XSS).",
        ],
        "recommandations": [
            "Mettre en place un pare-feu applicatif (WAF).",
            "Valider et filtrer les entrées utilisateur côté serveur.",
            "Appliquer les correctifs de sécurité des applications web.",
        ],
    },
    "Botnet / Infiltration": {
        "risque": "Élevé",
        "causes": [
            "Flux de longue durée avec peu de données échangées.",
            "Communication périodique vers un hôte externe.",
            "Comportement compatible avec un canal de commande.",
        ],
        "recommandations": [
            "Isoler la machine suspectée du réseau.",
            "Lancer une analyse antivirus et anti-malware complète.",
            "Bloquer les communications vers le serveur de commande.",
        ],
    },
    "Anomalie indéterminée": {
        "risque": "Moyen",
        "causes": [
            "Le comportement s'écarte du trafic normal appris.",
            "Aucune signature d'attaque connue clairement identifiée.",
            "Une analyse manuelle est recommandée.",
        ],
        "recommandations": [
            "Analyser manuellement les flux concernés.",
            "Vérifier les journaux système sur la période visée.",
            "Renforcer la surveillance du segment réseau touché.",
        ],
    },
}


def _col(df, nom, defaut=0.0):
    """Récupère une colonne si elle existe, sinon une série de valeurs par défaut."""
    if nom in df.columns:
        return pd.to_numeric(df[nom], errors="coerce").fillna(defaut)
    return pd.Series(defaut, index=df.index, dtype=float)


# Correspondance étiquettes CIC-IDS2017 -> familles d'attaques du catalogue
MAP_FAMILLE = {
    "ddos": "DDoS / DoS", "dos hulk": "DDoS / DoS", "dos goldeneye": "DDoS / DoS",
    "dos slowloris": "DDoS / DoS", "dos slowhttptest": "DDoS / DoS",
    "portscan": "PortScan",
    "bot": "Botnet / Infiltration", "infiltration": "Botnet / Infiltration",
    "ftp-patator": "Brute Force (FTP/SSH)", "ssh-patator": "Brute Force (FTP/SSH)",
    "web attack brute force": "Attaque Web", "web attack xss": "Attaque Web",
    "web attack sql injection": "Attaque Web", "web attack": "Attaque Web",
    "heartbleed": "Attaque Web",
}


def _famille_depuis_label(label):
    """Convertit une étiquette CIC-IDS2017 en famille du catalogue."""
    l = str(label).strip().lower()
    if l in ("benign", "normal"):
        return None
    if l in MAP_FAMILLE:
        return MAP_FAMILLE[l]
    if "portscan" in l:
        return "PortScan"
    if "ddos" in l or l.startswith("dos"):
        return "DDoS / DoS"
    if "patator" in l:
        return "Brute Force (FTP/SSH)"
    if "web" in l:
        return "Attaque Web"
    if "bot" in l or "infiltration" in l:
        return "Botnet / Infiltration"
    return "Anomalie indéterminée"


def _types_par_heuristique(d):
    """Règles d'expert appliquées aux caractéristiques du flux (repli si pas de Label)."""
    duree = _col(d, "Flow Duration")
    paquets_s = _col(d, "Flow Packets/s")
    fwd = _col(d, "Total Fwd Packets")
    bwd = _col(d, "Total Backward Packets")
    syn = _col(d, "SYN Flag Count")
    port = _col(d, "Destination Port", -1)
    taille = _col(d, "Packet Length Mean")

    types = np.full(len(d), "Anomalie indéterminée", dtype=object)

    web = port.isin([80, 443, 8080])
    admin = port.isin([21, 22, 23])

    m_botnet = (duree > duree.quantile(0.75)) & (fwd + bwd < 10)
    types[m_botnet.values] = "Botnet / Infiltration"

    m_web = web & (taille > 0)
    types[m_web.values] = "Attaque Web"

    m_brute = admin
    types[m_brute.values] = "Brute Force (FTP/SSH)"

    m_ddos = (paquets_s > paquets_s.quantile(0.75)) & (fwd > fwd.median())
    types[m_ddos.values] = "DDoS / DoS"

    m_scan = (duree < duree.quantile(0.25)) & (bwd <= 1) & (syn >= 0)
    types[m_scan.values] = "PortScan"

    return types


def diagnostiquer(df_anomalies, colonne_label="Label"):
    """
    Diagnostic : renvoie un DataFrame avec une colonne 'Type probable'
    et un dictionnaire de synthèse du type dominant (utilisé par le Dashboard).

    Utilise la VRAIE étiquette (colonne Label) si elle est présente dans le
    fichier analysé -> résultat exact, cohérent avec familles_detectees().
    Sinon, retombe sur des règles heuristiques appliquées aux caractéristiques
    du flux (comportement approximatif, utilisé seulement si aucun Label n'est
    disponible dans le fichier importé).
    """
    if len(df_anomalies) == 0:
        return df_anomalies.assign(**{"Type probable": []}), None

    d = df_anomalies.copy()
    d.columns = d.columns.str.strip()

    if colonne_label in d.columns:
        types = (
            d[colonne_label]
            .map(_famille_depuis_label)
            .fillna("Anomalie indéterminée")
            .values
        )
    else:
        types = _types_par_heuristique(d)

    d["Type probable"] = types

    dominant = pd.Series(types).value_counts().idxmax()
    info = CATALOGUE.get(dominant, CATALOGUE["Anomalie indéterminée"])
    synthese = {
        "type": dominant,
        "risque": info["risque"],
        "causes": info["causes"],
        "recommandations": info.get("recommandations", []),
        "repartition": pd.Series(types).value_counts().to_dict(),
    }
    return d, synthese


def familles_detectees(df_anomalies, colonne_label="Label"):
    """
    Renvoie la liste des familles d'attaques présentes parmi les flux suspects,
    chacune avec son nombre, son risque, ses causes et ses recommandations.

    Utilise la VRAIE étiquette (colonne Label) si elle existe -> le diagnostic
    varie donc selon le fichier. Sinon, retombe sur les règles heuristiques.
    Cohérent avec diagnostiquer() : les deux fonctions utilisent désormais la
    même source (Label si présent, sinon heuristique).
    """
    if len(df_anomalies) == 0:
        return []

    d = df_anomalies.copy()
    d.columns = d.columns.str.strip()

    if colonne_label in d.columns:
        familles = d[colonne_label].map(_famille_depuis_label)
    else:
        detail, _ = diagnostiquer(d, colonne_label=colonne_label)
        familles = detail["Type probable"]

    familles = familles.dropna()
    if len(familles) == 0:
        return []

    resultat = []
    for typ, n in familles.value_counts().items():
        info = CATALOGUE.get(typ, CATALOGUE["Anomalie indéterminée"])
        resultat.append({
            "type": typ,
            "count": int(n),
            "risque": info["risque"],
            "causes": info["causes"],
            "recommandations": info.get("recommandations", []),
        })
    return resultat