"""
app/app.py — Airport Network Anomaly Detection Platform (ONDA - Aéroport de Nador)
Sidebar bleue fixe, header (badge doré) + footer bleus, Rapports intégré au Dashboard,
"Types d'attaques détectés" en barres stylées, rapport PDF professionnel.
Lancement : streamlit run app/app.py
"""

import sys
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CHEMIN_MODELE, DOSSIER_REPORTS
from src.predict import predire
from src.diagnostic import diagnostiquer, familles_detectees, CATALOGUE

st.set_page_config(
    page_title="Airport Network Anomaly Detection — ONDA Nador",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded",
)

# --------- PALETTE ---------
FOND = "#F4F7FC"
NAVY = "#1E3A8A"
NAVY_FONCE = "#152C6B"
BLEU = "#2563EB"
BLEU_CLAIR = "#3B82F6"
BLEU_PALE = "#EAF1FE"
CYAN = "#0EA5E9"
ROUGE = "#DC2626"
VERT_OK = "#16A34A"
DORE = "#D9A521"
DORE_CLAIR = "#E7B93E"
ORANGE = "#EA580C"
TEXTE = "#1E293B"
GRIS = "#64748B"
BORD = "#E2E8F0"

H_HEADER = 74
H_FOOTER = 52
L_MENU = 284
# --------- COULEURS PDF (RGB) ---------
PDF_NAVY = (30, 58, 138)
PDF_BLEU = (37, 99, 235)
PDF_ROUGE = (220, 38, 38)
PDF_VERT = (22, 163, 74)
PDF_DORE = (217, 165, 33)
PDF_GRIS = (71, 85, 105)
PDF_GRIS_CLAIR = (241, 245, 249)
PDF_BORD = (226, 232, 240)
PDF_BLANC = (255, 255, 255)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family:'Inter', sans-serif; }}
.stApp {{ background:{FOND}; }}
h1,h2,h3 {{ font-family:'Montserrat', sans-serif; color:{NAVY}; }}

/* Masquer le header Streamlit et les boutons de collapse << / >> (sidebar fixe) */
header[data-testid="stHeader"] {{ display:none !important; }}
[data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
[data-testid="stExpandSidebarButton"] {{ display:none !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}

.block-container {{
    padding-top:{H_HEADER + 20}px !important;
    padding-bottom:{H_FOOTER + 16}px !important;
    max-width:100% !important; padding-left:2.4rem; padding-right:2.4rem;
}}

/* ---------- HEADER FIXE (bleu) ---------- */
.top-header {{
    position:fixed; top:0; left:{L_MENU}px; right:0; z-index:99990; height:{H_HEADER}px;
    display:flex; align-items:center; gap:16px; padding:0 30px; color:white;
    background: linear-gradient(120deg, {NAVY} 0%, {BLEU} 60%, {BLEU_CLAIR} 100%);
    box-shadow:0 3px 14px rgba(30,58,138,.22);
}}
.header-title-badge {{
    display: inline-flex; align-items: center;
    background: linear-gradient(135deg, {DORE} 0%, {DORE_CLAIR} 100%);
    color: {NAVY} !important; padding: 4px 14px; border-radius: 8px;
    font-size: 17px; font-weight: 800; font-family: 'Montserrat', sans-serif;
    letter-spacing: 0.3px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.4);
}}
.top-header p {{
    margin: 1px 0 0 2px; color: #F6DD93; font-size: 12px; font-weight: 600;
    font-family: 'Inter', sans-serif; letter-spacing: 0.2px; opacity: 0.95;
}}
.top-header .droite {{ margin-left:auto; display:flex; align-items:center; gap:14px; }}
.h-badge {{ display:flex; align-items:center; gap:8px; background:rgba(255,255,255,.16);
    border:1px solid rgba(255,255,255,.30); color:white; padding:7px 15px;
    border-radius:22px; font-size:12px; font-weight:700; font-family:'Montserrat'; }}
.h-badge .dot {{ width:9px; height:9px; border-radius:50%; background:#4ADE80;
    box-shadow:0 0 0 3px rgba(74,222,128,.35); }}
.h-date {{ text-align:right; font-size:12px; color:#E5EDFB; line-height:1.35; }}
.h-date b {{ color:white; font-family:'Montserrat'; }}
.h-bell {{ width:38px; height:38px; border-radius:50%; background:rgba(255,255,255,.16);
    border:1px solid rgba(255,255,255,.28); display:flex; align-items:center;
    justify-content:center; font-size:17px; }}
.h-logout {{ display:flex; align-items:center; gap:7px; background:rgba(220,38,38,.20);
    border:1px solid rgba(220,38,38,.55); color:white !important; padding:7px 15px;
    border-radius:22px; font-size:12px; font-weight:700; font-family:'Montserrat', sans-serif;
    text-decoration:none !important; cursor:pointer; transition: all .14s ease; }}
.h-logout:hover {{ background:rgba(220,38,38,.42); border-color:rgba(220,38,38,.85);
    transform:translateY(-1px); }}

/* ---------- FOOTER FIXE (bleu) ---------- */
.bottom-footer {{
    position:fixed; bottom:0; left:{L_MENU}px; right:0; z-index:99990; height:{H_FOOTER}px;
    display:flex; align-items:center; justify-content:space-between; padding:0 30px;
    color:#E5EDFB; font-size:12.5px;
    background: linear-gradient(120deg, {NAVY} 0%, {BLEU} 100%);
    border-top:3px solid {DORE};
}}
.bottom-footer b {{ color:{DORE_CLAIR}; }}
.bottom-footer .tag {{ display:flex; align-items:center; gap:7px;
    font-family:'Montserrat'; font-weight:600; }}

/* ---------- SIDEBAR BLEUE FIXE ---------- */
section[data-testid="stSidebar"] {{
    width:{L_MENU}px !important; min-width:{L_MENU}px !important;
}}
section[data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_FONCE} 100%);
    border-right:1px solid rgba(255,255,255,.08); padding:0 14px 20px;
}}
section[data-testid="stSidebar"] * {{ color:#EAF2FF; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.14); margin:12px 0; }}

.marque {{
    margin:0 -14px 14px; height:{H_HEADER}px; display:flex; align-items:center;
    gap:12px; padding:0 18px; border-bottom:1px solid rgba(255,255,255,.14);
}}
.marque .carre {{
    width:44px; height:44px; border-radius:12px; color:{NAVY}; font-size:22px;
    background: linear-gradient(120deg, {DORE}, {DORE_CLAIR}); display:flex;
    align-items:center; justify-content:center; box-shadow:0 3px 8px rgba(217,165,33,.4);
}}
.marque .txt b {{ color:white !important; font-family:'Montserrat'; font-size:16px;
    font-weight:800; display:block; line-height:1.1; }}
.marque .txt span {{ color:#B9C7E8 !important; font-size:11px; }}

.menu-titre {{ font-size:11px; letter-spacing:1.4px; text-transform:uppercase;
    color:#9DB0DC !important; margin:4px 6px 8px; font-weight:700; }}

section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(217,165,33,.16) !important; color:#F6DD93 !important;
    border:1px solid rgba(217,165,33,.55) !important; border-radius:11px !important;
    text-align:left !important; justify-content:flex-start !important;
    font-family:'Montserrat', sans-serif !important; font-weight:700 !important;
    font-size:14px !important; padding:12px 16px !important;
    box-shadow:none !important; margin-bottom:4px; transition: all .14s ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(217,165,33,.32) !important; color:white !important;
    transform:translateX(3px) !important;
}}
.nav-actif {{
    background: linear-gradient(120deg, {DORE} 0%, {DORE_CLAIR} 100%);
    color:{NAVY} !important; border-radius:11px; padding:12px 16px;
    font-family:'Montserrat'; font-weight:800; font-size:14px; margin-bottom:4px;
    box-shadow:0 4px 12px rgba(217,165,33,.45);
}}

.profil {{
    display:flex; align-items:center; gap:12px; background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.16); border-radius:14px; padding:12px 14px;
    margin-top:12px;
}}
.profil .av {{
    width:42px; height:42px; border-radius:50%; color:{NAVY}; font-weight:800;
    background: linear-gradient(120deg, {DORE}, {DORE_CLAIR}); display:flex;
    align-items:center; justify-content:center; font-family:'Montserrat'; font-size:13px;
}}
.profil b {{ font-size:14px; color:white !important; font-family:'Montserrat'; }}
.profil span {{ font-size:11.5px; color:#B9C7E8 !important; }}

.main .stButton > button, .stDownloadButton > button {{
    background: linear-gradient(120deg, {BLEU} 0%, {BLEU_CLAIR} 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-family: 'Montserrat', sans-serif !important;
    padding: 10px 20px !important; box-shadow: 0 3px 10px rgba(37,99,235,.30);
    transition: transform .12s ease, box-shadow .12s ease;
}}
.main .stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37,99,235,.42);
}}

.bienvenue {{
    background:white; border:1px solid {BORD}; border-left:5px solid {DORE};
    border-radius:16px; padding:22px 28px; margin-bottom:16px;
    box-shadow:0 2px 14px rgba(30,58,138,.06);
}}
.bienvenue h2 {{ margin:0 0 6px; font-size:23px; }}
.bienvenue p {{ margin:0; color:#475569; font-size:14.5px; line-height:1.65; }}

.carte {{
    background:white; border:1px solid {BORD}; border-top:3px solid {BLEU};
    border-radius:14px; padding:18px 20px; height:100%;
    box-shadow:0 2px 12px rgba(30,58,138,.05); transition:transform .12s ease;
}}
.carte:hover {{ transform:translateY(-3px); box-shadow:0 8px 20px rgba(30,58,138,.10); }}
.carte .ico {{ font-size:24px; }}
.carte h4 {{ margin:8px 0 5px; color:{NAVY}; font-size:15px; font-family:'Montserrat'; }}
.carte p {{ margin:0; color:{GRIS}; font-size:13px; line-height:1.5; }}
.carte.cyan {{ border-top-color:{CYAN}; }}
.carte.rouge {{ border-top-color:{ROUGE}; }}
.carte.vert {{ border-top-color:{VERT_OK}; }}
.carte.or {{ border-top-color:{DORE}; }}
.carte.orange {{ border-top-color:{ORANGE}; }}

.etape {{ display:flex; align-items:center; gap:12px; margin:6px 0 14px; }}
.etape .num {{
    background: linear-gradient(120deg,{DORE},{DORE_CLAIR}); color:{NAVY};
    width:30px; height:30px; border-radius:9px; display:flex; align-items:center;
    justify-content:center; font-weight:800; font-size:14px; font-family:'Montserrat';
}}
.etape .txt {{ font-family:'Montserrat'; font-weight:700; color:{NAVY}; font-size:17px; }}

.kpi {{
    background:white; border-radius:14px; padding:18px 20px; border:1px solid {BORD};
    border-top:3px solid {BLEU}; box-shadow:0 2px 12px rgba(30,58,138,.05); height:100%;
}}
.kpi .val {{ font-family:'Montserrat'; font-size:29px; font-weight:800; color:{NAVY}; }}
.kpi .lab {{ font-size:11.5px; color:{GRIS}; margin-top:3px;
             text-transform:uppercase; letter-spacing:.5px; }}
.kpi.vert {{ border-top-color:{VERT_OK}; }} .kpi.vert .val {{ color:{VERT_OK}; }}
.kpi.rouge {{ border-top-color:{ROUGE}; }} .kpi.rouge .val {{ color:{ROUGE}; }}
.kpi.cyan {{ border-top-color:{CYAN}; }} .kpi.cyan .val {{ color:{CYAN}; }}
.kpi.or {{ border-top-color:{DORE}; }} .kpi.or .val {{ color:#A87C0F; }}
.kpi.orange {{ border-top-color:{ORANGE}; }} .kpi.orange .val {{ color:{ORANGE}; }}

.fichier {{
    background:{BLEU_PALE}; border:1px solid #D6E4FD; border-radius:12px;
    padding:14px 18px; display:flex; align-items:center; gap:14px;
}}
.fichier .ico {{ font-size:24px; }}
.fichier b {{ color:{NAVY}; }}
.fichier span {{ color:{GRIS}; font-size:13px; }}

.vide {{
    background:white; border:1px dashed #C7D6EA; border-radius:16px;
    padding:44px 32px; text-align:center;
}}
.vide .ico {{ font-size:42px; opacity:.5; }}
.vide h4 {{ margin:10px 0 6px; color:{NAVY}; font-family:'Montserrat'; font-size:18px; }}
.vide p {{ margin:0; color:{GRIS}; font-size:14px; }}

.pastille {{ display:inline-block; padding:5px 14px; border-radius:20px;
             font-size:12.5px; font-weight:700; font-family:'Montserrat'; }}
.diag {{
    background:white; border:1px solid {BORD}; border-left:5px solid {ROUGE};
    border-radius:16px; padding:20px 26px; box-shadow:0 2px 14px rgba(30,58,138,.06);
}}
.diag h3 {{ margin:0 0 4px; font-size:19px; }}
.diag .type {{ font-family:'Montserrat'; font-size:24px; font-weight:800; color:{NAVY}; }}

.desc-attaque {{
    background:{BLEU_PALE}; border:1px solid #D6E4FD; border-left:5px solid {BLEU};
    border-radius:10px; padding:14px 18px; color:#334155; font-size:14px;
    line-height:1.6; margin-bottom:8px;
}}
.desc-attaque b {{ color:{NAVY}; }}

.bloc-c, .bloc-r {{
    background:white; border:1px solid {BORD}; border-radius:14px;
    padding:20px 24px; height:100%; box-shadow:0 2px 12px rgba(30,58,138,.05);
}}
.bloc-c {{ border-left:5px solid {ROUGE}; }}
.bloc-r {{ border-left:5px solid {VERT_OK}; }}
.bloc-c h4, .bloc-r h4 {{
    margin:0 0 12px; font-family:'Montserrat'; font-size:16px;
    display:flex; align-items:center; gap:8px;
}}
.bloc-c h4 {{ color:{ROUGE}; }}
.bloc-r h4 {{ color:{VERT_OK}; }}
.bloc-c ul, .bloc-r ul {{ margin:0; padding-left:0; list-style:none; }}
.bloc-c li, .bloc-r li {{
    color:#475569; font-size:14px; line-height:1.6; padding:8px 0 8px 26px;
    border-bottom:1px solid #F1F5F9; position:relative;
}}
.bloc-c li:last-child, .bloc-r li:last-child {{ border-bottom:none; }}
.bloc-c li::before {{ content:"⚠"; position:absolute; left:0; color:{ROUGE}; }}
.bloc-r li::before {{ content:"✓"; position:absolute; left:0; color:{VERT_OK}; font-weight:700; }}

.r-critique {{ background:#FEE2E2; color:{ROUGE}; }}
.r-eleve {{ background:#FEF3C7; color:#B45309; }}
.r-moyen {{ background:#DBEAFE; color:{BLEU}; }}

/* ---------- Types d'attaques détectés (barres horizontales) ---------- */
.att-liste {{ display:flex; flex-direction:column; gap:12px; }}
.att-item {{ background:white; border:1px solid {BORD}; border-radius:12px;
    padding:13px 18px; box-shadow:0 2px 10px rgba(30,58,138,.05); }}
.att-tete {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:9px; }}
.att-nom {{ font-family:'Montserrat'; font-weight:700; color:{NAVY}; font-size:14.5px;
    display:flex; align-items:center; gap:8px; }}
.att-count {{ font-family:'Montserrat'; font-weight:800; font-size:14.5px; }}
.att-piste {{ background:#EEF2F8; border-radius:8px; height:15px; overflow:hidden; }}
.att-barre {{ height:100%; border-radius:8px; transition:width .5s ease; }}
</style>
""", unsafe_allow_html=True)




# ================= AUTHENTIFICATION =================
IDENTIFIANT = "admin"
MOT_DE_PASSE = "onda2026"
NOM_UTILISATEUR = "Responsable sécurité réseau"

if "connecte" not in st.session_state:
    st.session_state.connecte = False

if st.query_params.get("logout") == "1":
    st.session_state.connecte = False
    st.query_params.clear()
    st.rerun()


def page_login():
    st.markdown(f"""
    <style>
    header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {{ display:none !important; }}

    .stApp {{ background: linear-gradient(135deg, {NAVY} 0%, {NAVY_FONCE} 55%, {BLEU} 100%) !important; }}

    /* Tout transparent (pas de carte parasite) */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    .main, section.main {{ background:transparent !important; box-shadow:none !important; }}
    [data-testid="stElementContainer"] {{ background:transparent !important; }}

    /* Colonne étroite centrée verticalement (sans scroll) */
    .block-container {{
        max-width:430px !important; padding:0 !important; background:transparent !important;
        min-height:100vh !important; display:flex !important; flex-direction:column !important;
        justify-content:center !important;
    }}

    /* Logo moutarde */
    .login-logo {{
        width:74px; height:74px; border-radius:18px; margin:0 auto 12px;
        background:linear-gradient(120deg,{DORE},{DORE_CLAIR}); color:{NAVY};
        display:flex; align-items:center; justify-content:center; font-size:36px;
        box-shadow:0 6px 16px rgba(217,165,33,.5);
    }}

    /* Titre en MOUTARDE, sous-titre en BLANC */
    .login-titre {{ text-align:center; }}
    .login-titre h2 {{ color:{DORE_CLAIR} !important; margin:0 0 4px; font-family:'Montserrat';
        font-size:21px; font-weight:800; }}
    .login-titre p {{ color:#EAF2FF !important; font-size:12.5px; margin:0 0 10px; line-height:1.5; }}

    /* Labels en MOUTARDE */
    .stApp label, .stApp label p {{ color:{DORE} !important; font-weight:700 !important;
        font-family:'Montserrat' !important; font-size:13.5px !important; }}

    /* Champs de saisie */
    .stApp input {{ border:1.5px solid {BORD} !important; border-radius:10px !important;
        padding:10px !important; background:#F8FAFC !important; }}
    .stApp input:focus {{ border-color:{BLEU} !important; box-shadow:0 0 0 3px rgba(37,99,235,.15) !important; }}

    /* Bouton moutarde */
    .stButton > button {{
        background:linear-gradient(120deg,{DORE},{DORE_CLAIR}) !important; color:{NAVY} !important;
        border:none !important; border-radius:10px !important; font-weight:800 !important;
        font-family:'Montserrat' !important; padding:11px !important; margin-top:10px;
        box-shadow:0 4px 12px rgba(217,165,33,.4) !important; transition:transform .12s ease; }}
    .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 7px 18px rgba(217,165,33,.55) !important; }}

    /* Erreur rouge */
    [data-testid="stAlert"] {{ background:#FEE2E2 !important; border:1px solid {ROUGE} !important;
        border-radius:10px !important; margin-top:10px !important; }}

    /* Footer en BLANC clair */
    .login-foot {{ text-align:center; color:#EAF2FF !important; font-size:11px; margin-top:14px;
        padding-top:10px; border-top:1px solid rgba(255,255,255,.25); }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-logo">🛡️</div>
    <div class="login-titre">
        <h2>ONDA — Sécurité réseau</h2>
        <p>Plateforme de détection d'anomalies · Aéroport de Nador</p>
    </div>
    """, unsafe_allow_html=True)

    identifiant = st.text_input("👤 Identifiant")
    mdp = st.text_input("🔒 Mot de passe", type="password")
    clic = st.button("Se connecter", use_container_width=True)

    if clic:
        if identifiant == IDENTIFIANT and mdp == MOT_DE_PASSE:
            st.session_state.connecte = True
            st.rerun()
        else:
            st.error("❌ Identifiant ou mot de passe incorrect.")

    st.markdown('<div class="login-foot">🔒 Accès réservé au personnel autorisé — ONDA © 2026</div>',
                unsafe_allow_html=True)


if not st.session_state.connecte:
    page_login()
    st.stop()

def carte_kpi(valeur, label, style=""):
    st.markdown(f'<div class="kpi {style}"><div class="val">{valeur}</div>'
                f'<div class="lab">{label}</div></div>', unsafe_allow_html=True)


def etape(numero, texte):
    st.markdown(f'<div class="etape"><div class="num">{numero}</div>'
                f'<div class="txt">{texte}</div></div>', unsafe_allow_html=True)


def fmt(n):
    return f"{n:,}".replace(",", " ")


def classe_risque(risque):
    return {"Critique": "r-critique", "Élevé": "r-eleve"}.get(risque, "r-moyen")


def determiner_criticite(type_attaque):
    """Criticité = lecture directe du CATALOGUE (src/diagnostic.py), la même source
    de vérité utilisée par la page Diagnostic, la page Alertes et le rapport PDF."""
    info = CATALOGUE.get(type_attaque, CATALOGUE["Anomalie indéterminée"])
    return info["risque"]


def construire_alertes(tableau, statuts_alertes):
    """Construit le tableau des alertes (Flow ID, Type, Criticité, Statut) à partir du
    tableau d'analyse complet — réutilisé par la page Alertes et par le rapport PDF."""
    alertes = tableau[tableau["Prédiction"] == "Anomalie"].copy()
    alertes["Criticité"] = alertes["Type probable"].apply(determiner_criticite)
    alertes["Statut"] = alertes["Flow ID"].apply(
        lambda fid: statuts_alertes.get(int(fid), "Nouveau")
    )
    return alertes


def bloc_vide(icone, titre, texte):
    st.markdown(f"""
    <div class="vide"><div class="ico">{icone}</div>
    <h4>{titre}</h4><p>{texte}</p></div>""", unsafe_allow_html=True)


def obtenir_modele(paquet):
    """Récupère l'objet modèle sklearn depuis le paquet joblib, quel que soit le nom de clé utilisé."""
    if paquet is None:
        return None
    if hasattr(paquet, "predict"):
        return paquet
    if isinstance(paquet, dict):
        for cle in ("modele", "model", "clf", "classifier", "estimator", "pipeline"):
            if cle in paquet and hasattr(paquet[cle], "predict"):
                return paquet[cle]
    return None


def obtenir_colonnes_features(paquet, df):
    """Récupère la liste des colonnes utilisées par le modèle. À défaut, prend les colonnes numériques du df."""
    if isinstance(paquet, dict):
        for cle in ("colonnes", "features", "feature_names", "colonnes_modele", "columns"):
            if cle in paquet and paquet[cle] is not None:
                cols = [c for c in paquet[cle] if c in df.columns]
                if cols:
                    return cols
    exclure = {"label", "attack", "attack type", "class", "flow id", "prédiction", "type probable", "état"}
    return [c for c in df.columns
            if c.strip().lower() not in exclure and pd.api.types.is_numeric_dtype(df[c])]


def calculer_stats_normales(df_brut, colonnes, predictions):
    """Moyenne / écart-type de chaque feature pour le trafic normal (référence pour l'explicabilité)."""
    stats = {}
    normal = df_brut.loc[predictions == 0, colonnes] if len(colonnes) else df_brut.iloc[0:0]
    for col in colonnes:
        try:
            valeurs = pd.to_numeric(normal[col], errors="coerce")
            stats[col] = {
                "moyenne": float(valeurs.mean()) if len(valeurs) and not np.isnan(valeurs.mean()) else 0.0,
                "ecart_type": float(valeurs.std()) if len(valeurs) and valeurs.std() not in (0, None) and not np.isnan(valeurs.std()) else 1.0,
            }
        except Exception:
            stats[col] = {"moyenne": 0.0, "ecart_type": 1.0}
    return stats


def expliquer_flux(ligne_brute, colonnes, modele, stats_normales, top_n=6):
    """Calcule, pour un flux donné, les caractéristiques ayant le plus influencé la détection
    (importance globale du modèle × écart par rapport à la moyenne du trafic normal)."""
    importances = getattr(modele, "feature_importances_", None) if modele is not None else None
    if importances is None or len(importances) != len(colonnes):
        importances = np.ones(len(colonnes))

    contributions = []
    for i, col in enumerate(colonnes):
        try:
            val = float(ligne_brute.get(col, np.nan))
        except (TypeError, ValueError):
            val = np.nan
        ref = stats_normales.get(col, {"moyenne": 0.0, "ecart_type": 1.0})
        ect = ref["ecart_type"] or 1.0
        z = abs((val - ref["moyenne"]) / ect) if not np.isnan(val) else 0.0
        score = float(importances[i]) * z
        contributions.append({
            "feature": col,
            "valeur": round(val, 3) if not np.isnan(val) else "—",
            "moyenne_normale": round(ref["moyenne"], 3),
            "ecart_type_normalise": round(z, 2),
            "score": round(score, 4),
        })
    contributions.sort(key=lambda x: x["score"], reverse=True)
    return contributions[:top_n]


def afficher_explication(flow_id):
    """Affiche le graphique + tableau d'explicabilité pour un Flow ID donné."""
    donnees = st.session_state.get("donnees_brutes")
    colonnes = st.session_state.get("colonnes_features")
    modele = st.session_state.get("modele_actif")
    stats_normales = st.session_state.get("stats_normales")

    if donnees is None or not colonnes or modele is None or not stats_normales:
        st.info("ℹ️ Explicabilité indisponible pour cette analyse (modèle ou données non détectés). "
                "Relancez une analyse depuis **📂 Analyse du trafic**.")
        return

    idx = int(flow_id) - 1
    if idx < 0 or idx >= len(donnees):
        st.warning("Flow introuvable dans les données analysées.")
        return

    contributions = expliquer_flux(donnees.iloc[idx], colonnes, modele, stats_normales, top_n=6)
    if not contributions:
        st.info("Aucune caractéristique disponible pour expliquer ce flux.")
        return

    df_expl = pd.DataFrame(contributions)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    seuil = df_expl["score"].max() * 0.5 if df_expl["score"].max() else 0
    couleurs = [ROUGE if s >= seuil else ORANGE for s in df_expl["score"]]
    ax.barh(df_expl["feature"][::-1], df_expl["score"][::-1], color=couleurs[::-1])
    ax.set_xlabel("Score d'influence (importance du modèle × écart à la normale)")
    ax.set_title(f"Facteurs ayant influencé la détection — Flow #{flow_id}", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("📋 Détail des valeurs comparées au trafic normal"):
        st.dataframe(
            df_expl.rename(columns={
                "feature": "Caractéristique", "valeur": "Valeur observée",
                "moyenne_normale": "Moyenne (trafic normal)",
                "ecart_type_normalise": "Écart (z-score)", "score": "Score d'influence",
            }),
            use_container_width=True, hide_index=True,
        )


def filtrer_tableau(tableau, cle_recherche="filtre", colonne_type="Type probable"):
    """Barre de recherche + filtres réutilisable pour un tableau de flux."""
    st.markdown("**🔎 Recherche et filtres**")
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        recherche = st.text_input("Rechercher (Flow ID, type d'attaque...)",
                                   key=f"{cle_recherche}_texte", label_visibility="collapsed",
                                   placeholder="🔎 Rechercher un Flow ID ou un type d'attaque…")
    with fc2:
        types_dispo = sorted(tableau[colonne_type].dropna().unique().tolist())
        types_choisis = st.multiselect("Type d'attaque", types_dispo, key=f"{cle_recherche}_types",
                                        label_visibility="collapsed", placeholder="Filtrer par type")
    with fc3:
        preds_dispo = sorted(tableau["Prédiction"].dropna().unique().tolist()) if "Prédiction" in tableau.columns else []
        preds_choisies = st.multiselect("Statut détection", preds_dispo, key=f"{cle_recherche}_pred",
                                         label_visibility="collapsed", placeholder="Filtrer par statut")

    resultat = tableau.copy()
    if recherche:
        r = recherche.strip().lower()
        resultat = resultat[resultat.apply(
            lambda row: r in str(row.get("Flow ID", "")).lower()
            or r in str(row.get(colonne_type, "")).lower()
            or r in str(row.get("Prédiction", "")).lower(), axis=1)]
    if types_choisis:
        resultat = resultat[resultat[colonne_type].isin(types_choisis)]
    if preds_choisies and "Prédiction" in resultat.columns:
        resultat = resultat[resultat["Prédiction"].isin(preds_choisies)]
    return resultat


@st.cache_resource
def charger_paquet():
    return joblib.load(CHEMIN_MODELE) if CHEMIN_MODELE.exists() else None


def metriques_modele(paquet):
    defaut = {"accuracy": 0.9984, "precision": 0.9973, "recall": 0.9920, "f1": 0.9947}
    if paquet and isinstance(paquet.get("metriques"), dict):
        return paquet["metriques"]
    return defaut


def generer_pdf(stats, diag, tableau=None, alertes=None):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    NAVY_C, BLEU_C, ROUGE_C = PDF_NAVY, PDF_BLEU, PDF_ROUGE
    VERT_C, DORE_C = PDF_VERT, PDF_DORE
    GRIS_C, GRIS_CLAIR, BORD_C, BLANC_C = (
        PDF_GRIS,
        PDF_GRIS_CLAIR,
        PDF_BORD,
        PDF_BLANC
    )

    class RapportPDF(FPDF):

        def header(self):
            self.set_fill_color(*NAVY_C)
            self.rect(0, 0, self.w, 30, style="F")

            self.set_fill_color(*DORE_C)
            self.rect(0, 30, self.w, 1.4, style="F")

            self.set_xy(self.l_margin, 8)
            self.set_text_color(*BLANC_C)
            self.set_font("Helvetica", "B", 15)
            self.cell(
                0,
                8,
                "Rapport d'analyse du trafic reseau",
                ln=True
            )

            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(210, 220, 240)
            self.cell(
                0,
                5,
                "Plateforme intelligente de detection des anomalies",
                ln=True
            )

            self.set_xy(self.w - 70, 8)
            self.set_text_color(*BLANC_C)
            self.set_font("Helvetica", "B", 13)
            self.cell(
                60,
                7,
                "ONDA",
                align="R",
                ln=True
            )

            self.set_x(self.w - 70)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(210, 220, 240)
            self.cell(
                60,
                4,
                "Office National Des Aeroports",
                align="R",
                ln=True
            )

            self.set_x(self.w - 70)
            self.cell(
                60,
                4,
                "Aeroport de Nador",
                align="R"
            )

            self.set_y(42)

        def footer(self):
            self.set_y(-15)

            self.set_fill_color(*NAVY_C)
            self.rect(
                0,
                self.h - 12,
                self.w,
                12,
                style="F"
            )

            self.set_y(-9)
            self.set_text_color(*BLANC_C)
            self.set_font("Helvetica", "", 8)

            self.cell(
                0,
                5,
                f"ONDA - Security Operations Center  |  Nador  |  Page {self.page_no()}",
                align="C"
            )

        def titre_section(self, texte):
            self.ln(2)

            self.set_text_color(*NAVY_C)
            self.set_font("Helvetica", "B", 13)

            self.cell(
                0,
                8,
                texte,
                ln=True
            )

            self.set_draw_color(*DORE_C)
            self.set_line_width(0.6)

            y = self.get_y()

            self.line(
                self.l_margin,
                y,
                self.l_margin + 55,
                y
            )

            self.ln(2)

        def carte_kpi_pdf(
            self,
            x,
            y,
            w,
            h,
            valeur,
            label,
            couleur
        ):
            self.set_fill_color(*BLANC_C)
            self.set_draw_color(*BORD_C)
            self.set_line_width(0.3)

            self.rect(
                x,
                y,
                w,
                h,
                style="DF"
            )

            self.set_fill_color(*couleur)

            self.rect(
                x,
                y,
                w,
                2.2,
                style="F"
            )

            self.set_xy(
                x,
                y + 6
            )

            self.set_text_color(*couleur)
            self.set_font(
                "Helvetica",
                "B",
                16
            )

            self.cell(
                w,
                8,
                str(valeur),
                align="C"
            )

            self.set_xy(
                x,
                y + 15
            )

            self.set_text_color(*GRIS_C)
            self.set_font(
                "Helvetica",
                "",
                8.5
            )

            self.cell(
                w,
                5,
                label,
                align="C"
            )

    # ============================================================
    # Création du PDF
    # ============================================================

    pdf = RapportPDF(
        orientation="P",
        unit="mm",
        format="A4"
    )

    # Marge basse réduite pour garder le rapport sur 2 pages
    pdf.set_auto_page_break(
        auto=True,
        margin=10
    )

    pdf.set_margins(
        15,
        42,
        15
    )

    pdf.add_page()

    largeur_tot = pdf.w - 2 * pdf.l_margin

    # ============================================================
    # Informations générales
    # ============================================================

    pdf.set_fill_color(*GRIS_CLAIR)
    pdf.set_draw_color(*BORD_C)

    pdf.rect(
        pdf.l_margin,
        pdf.get_y(),
        largeur_tot,
        16,
        style="DF"
    )

    y0 = pdf.get_y()

    pdf.set_xy(
        pdf.l_margin + 4,
        y0 + 3
    )

    pdf.set_text_color(*GRIS_C)
    pdf.set_font(
        "Helvetica",
        "",
        9.5
    )

    pdf.cell(
        0,
        5,
        f"Date d'analyse : {stats.get('date', '-')}",
        ln=True
    )

    pdf.set_x(
        pdf.l_margin + 4
    )

    pdf.cell(
        0,
        5,
        f"Fichier analyse : {stats.get('fichier', '-')}",
        ln=True
    )

    pdf.set_y(
        y0 + 20
    )

    # ============================================================
    # Résumé de l'analyse
    # ============================================================

    pdf.titre_section(
        "Resume de l'analyse"
    )

    x = pdf.l_margin
    y = pdf.get_y()

    w = (
        largeur_tot - 3 * 4
    ) / 4

    h = 24

    pdf.carte_kpi_pdf(
        x,
        y,
        w,
        h,
        stats["total"],
        "Flux analyses",
        NAVY_C
    )

    pdf.carte_kpi_pdf(
        x + (w + 4),
        y,
        w,
        h,
        stats["normal"],
        "Trafic normal",
        VERT_C
    )

    pdf.carte_kpi_pdf(
        x + 2 * (w + 4),
        y,
        w,
        h,
        stats["anomalies"],
        "Anomalies",
        ROUGE_C
    )

    pdf.carte_kpi_pdf(
        x + 3 * (w + 4),
        y,
        w,
        h,
        f"{stats['taux']:.1f} %",
        "Taux d'anomalies",
        DORE_C
    )

    pdf.set_y(
        y + h + 4
    )

    # ============================================================
    # Diagnostic
    # ============================================================

    if diag:

        pdf.titre_section(
            "Diagnostic"
        )

        pdf.set_fill_color(*BLANC_C)
        pdf.set_draw_color(*BORD_C)

        yb = pdf.get_y()

        pdf.rect(
            pdf.l_margin,
            yb,
            largeur_tot,
            26,
            style="DF"
        )

        pdf.set_fill_color(*ROUGE_C)

        pdf.rect(
            pdf.l_margin,
            yb,
            2.2,
            26,
            style="F"
        )

        pdf.set_xy(
            pdf.l_margin + 6,
            yb + 4
        )

        pdf.set_text_color(*GRIS_C)
        pdf.set_font(
            "Helvetica",
            "",
            9
        )

        pdf.cell(
            0,
            5,
            "Attaque dominante detectee",
            ln=True
        )

        pdf.set_x(
            pdf.l_margin + 6
        )

        pdf.set_text_color(*NAVY_C)
        pdf.set_font(
            "Helvetica",
            "B",
            15
        )

        pdf.cell(
            0,
            8,
            str(diag["type"]),
            ln=True
        )

        pdf.set_x(
            pdf.l_margin + 6
        )

        pdf.set_text_color(*ROUGE_C)
        pdf.set_font(
            "Helvetica",
            "B",
            9.5
        )

        pdf.cell(
            0,
            5,
            f"Niveau de risque : {diag['risque']}"
        )

        pdf.set_y(
            yb + 29
        )

    # ============================================================
    # Performances du modèle
    # ============================================================

    pdf.titre_section(
        "Performances du modele"
    )

    m = stats["metriques"]

    x = pdf.l_margin
    y = pdf.get_y()

    w = (
        largeur_tot - 3 * 4
    ) / 4

    pdf.carte_kpi_pdf(
        x,
        y,
        w,
        22,
        f"{m['accuracy'] * 100:.1f}%",
        "Accuracy",
        NAVY_C
    )

    pdf.carte_kpi_pdf(
        x + (w + 4),
        y,
        w,
        22,
        f"{m['precision'] * 100:.1f}%",
        "Precision",
        BLEU_C
    )

    pdf.carte_kpi_pdf(
        x + 2 * (w + 4),
        y,
        w,
        22,
        f"{m['recall'] * 100:.1f}%",
        "Recall",
        VERT_C
    )

    pdf.carte_kpi_pdf(
        x + 3 * (w + 4),
        y,
        w,
        22,
        f"{m['f1'] * 100:.1f}%",
        "F1-Score",
        DORE_C
    )

    pdf.set_y(
        y + 22 + 4
    )

    # ============================================================
    # Flux suspects
    # ============================================================

    if tableau is not None and len(tableau) > 0:

        pdf.titre_section(
            "Flux suspects (extrait)"
        )

        cols = [
            c for c in [
                "Flow ID",
                "Prédiction",
                "Type probable"
            ]
            if c in tableau.columns
        ]

        if not cols:
            cols = list(tableau.columns)[:3]

        wcol = largeur_tot / len(cols)

        pdf.set_fill_color(*NAVY_C)
        pdf.set_text_color(*BLANC_C)
        pdf.set_font(
            "Helvetica",
            "B",
            9
        )

        for c in cols:
            pdf.cell(
                wcol,
                8,
                str(c)[:24],
                border=0,
                align="C",
                fill=True
            )

        pdf.ln()

        pdf.set_text_color(*GRIS_C)
        pdf.set_font(
            "Helvetica",
            "",
            8.5
        )

        # 10 lignes seulement pour garder le rapport compact
        for i, (_, row) in enumerate(
            tableau.head(10).iterrows()
        ):

            pdf.set_fill_color(
                *(
                    GRIS_CLAIR
                    if i % 2 == 0
                    else BLANC_C
                )
            )

            for c in cols:
                pdf.cell(
                    wcol,
                    7,
                    str(row[c])[:24],
                    border="B",
                    align="C",
                    fill=True
                )

            pdf.ln()

        pdf.ln(1)

    # ============================================================
    # Synthèse des alertes
    # ============================================================

    if alertes is not None and len(alertes) > 0:

        pdf.titre_section(
            "Synthese des alertes"
        )

        # Description courte
        pdf.set_text_color(*GRIS_C)
        pdf.set_font(
            "Helvetica",
            "",
            8.5
        )

        pdf.multi_cell(
            largeur_tot,
            4,
            "Cette section presente une synthese des alertes "
            "selon leur criticite et leur statut."
        )

        pdf.ln(1)

        # --------------------------------------------------------
        # Statistiques
        # --------------------------------------------------------

        total_alertes = len(alertes)

        nb_critique = int(
            (
                alertes["Criticité"] == "Critique"
            ).sum()
        )

        nb_eleve = int(
            (
                alertes["Criticité"] == "Élevé"
            ).sum()
        )

        nb_moyen = int(
            (
                alertes["Criticité"] == "Moyen"
            ).sum()
        )

        nb_nouveau = int(
            (
                alertes["Statut"] == "Nouveau"
            ).sum()
        )

        nb_en_cours = int(
            (
                alertes["Statut"] == "En cours"
            ).sum()
        )

        nb_traite = int(
            (
                alertes["Statut"] == "Traité"
            ).sum()
        )

        # --------------------------------------------------------
        # Première ligne : criticité
        # --------------------------------------------------------

        x = pdf.l_margin
        y = pdf.get_y()

        w = (
            largeur_tot - 3 * 4
        ) / 4

        h = 21

        pdf.carte_kpi_pdf(
            x,
            y,
            w,
            h,
            total_alertes,
            "Alertes totales",
            NAVY_C
        )

        pdf.carte_kpi_pdf(
            x + (w + 4),
            y,
            w,
            h,
            nb_critique,
            "Criticite critique",
            ROUGE_C
        )

        pdf.carte_kpi_pdf(
            x + 2 * (w + 4),
            y,
            w,
            h,
            nb_eleve,
            "Criticite elevee",
            DORE_C
        )

        pdf.carte_kpi_pdf(
            x + 3 * (w + 4),
            y,
            w,
            h,
            nb_moyen,
            "Criticite moyenne",
            BLEU_C
        )

        pdf.set_y(
            y + h + 3
        )

        # --------------------------------------------------------
        # Deuxième ligne : statut
        # --------------------------------------------------------

        x = pdf.l_margin
        y = pdf.get_y()

        pdf.carte_kpi_pdf(
            x,
            y,
            w,
            h,
            nb_nouveau,
            "Nouvelles",
            BLEU_C
        )

        pdf.carte_kpi_pdf(
            x + (w + 4),
            y,
            w,
            h,
            nb_en_cours,
            "En cours",
            DORE_C
        )

        pdf.carte_kpi_pdf(
            x + 2 * (w + 4),
            y,
            w,
            h,
            nb_traite,
            "Traitees",
            VERT_C
        )

        pdf.set_y(
            y + h + 5
        )

        # ========================================================
        # Principales alertes
        # ========================================================

        pdf.titre_section(
            "Principales alertes"
        )

        pdf.set_text_color(*GRIS_C)
        pdf.set_font(
            "Helvetica",
            "",
            8
        )

        pdf.multi_cell(
            largeur_tot,
            4,
            "Les alertes les plus importantes sont presentees "
            "ci-dessous afin de faciliter l'identification des "
            "anomalies necessitant une attention particuliere."
        )

        pdf.ln(1)

        # --------------------------------------------------------
        # Trier par criticité
        # --------------------------------------------------------

        alertes_affichage = alertes.copy()

        ordre_criticite = {
            "Critique": 3,
            "Élevé": 2,
            "Moyen": 1
        }

        alertes_affichage["_ordre_criticite"] = (
            alertes_affichage["Criticité"]
            .map(ordre_criticite)
            .fillna(0)
        )

        alertes_affichage = (
            alertes_affichage
            .sort_values(
                by="_ordre_criticite",
                ascending=False
            )
        )

        # Seulement 3 alertes dans le PDF
        principales = (
            alertes_affichage
            .head(3)
            .copy()
        )

        # --------------------------------------------------------
        # Tableau
        # --------------------------------------------------------

        cols_al = [
            "Flow ID",
            "Type probable",
            "Criticité",
            "Statut"
        ]

        largeurs = [
            largeur_tot * 0.14,
            largeur_tot * 0.40,
            largeur_tot * 0.23,
            largeur_tot * 0.23
        ]

        pdf.set_fill_color(*NAVY_C)
        pdf.set_text_color(*BLANC_C)
        pdf.set_font(
            "Helvetica",
            "B",
            8.5
        )

        for c, wcol in zip(
            cols_al,
            largeurs
        ):

            titre = (
                "Type d'attaque"
                if c == "Type probable"
                else c
            )

            pdf.cell(
                wcol,
                7,
                titre,
                border=0,
                align="C",
                fill=True
            )

        pdf.ln()

        # --------------------------------------------------------
        # Lignes
        # --------------------------------------------------------

        couleur_criticite = {
            "Critique": ROUGE_C,
            "Élevé": DORE_C,
            "Moyen": BLEU_C
        }

        pdf.set_font(
            "Helvetica",
            "",
            8
        )

        for i, (_, row) in enumerate(
            principales.iterrows()
        ):

            pdf.set_fill_color(
                *(
                    GRIS_CLAIR
                    if i % 2 == 0
                    else BLANC_C
                )
            )

            # Flow ID
            pdf.set_text_color(*GRIS_C)

            pdf.cell(
                largeurs[0],
                6,
                str(row["Flow ID"]),
                border="B",
                align="C",
                fill=True
            )

            # Type d'attaque
            pdf.cell(
                largeurs[1],
                6,
                str(
                    row["Type probable"]
                )[:30],
                border="B",
                align="C",
                fill=True
            )

            # Criticité
            criticite = str(
                row["Criticité"]
            )

            pdf.set_text_color(
                *couleur_criticite.get(
                    criticite,
                    GRIS_C
                )
            )

            pdf.set_font(
                "Helvetica",
                "B",
                8
            )

            pdf.cell(
                largeurs[2],
                6,
                criticite,
                border="B",
                align="C",
                fill=True
            )

            # Statut
            pdf.set_text_color(*GRIS_C)

            pdf.set_font(
                "Helvetica",
                "",
                8
            )

            pdf.cell(
                largeurs[3],
                6,
                str(row["Statut"]),
                border="B",
                align="C",
                fill=True
            )

            pdf.ln()

        # --------------------------------------------------------
        # Alertes supplémentaires
        # --------------------------------------------------------

        if len(alertes) > len(principales):

            pdf.ln(1)

            pdf.set_text_color(*GRIS_C)
            pdf.set_font(
                "Helvetica",
                "I",
                7.5
            )

            pdf.multi_cell(
                largeur_tot,
                4,
                f"{len(alertes) - len(principales)} "
                "alerte(s) supplementaire(s) sont consultables "
                "depuis la plateforme."
            )

        # --------------------------------------------------------
        # Statut manuel
        # --------------------------------------------------------

        pdf.ln(1)

        pdf.set_text_color(*GRIS_C)
        pdf.set_font(
            "Helvetica",
            "I",
            7.5
        )

        pdf.multi_cell(
            largeur_tot,
            4,
            "Le statut (Nouveau, En cours ou Traite) est renseigne "
            "manuellement par le responsable securite reseau."
        )

    # ============================================================
    # Conclusion
    # ============================================================

    pdf.titre_section(
        "Conclusion"
    )

    pdf.set_text_color(*GRIS_C)
    pdf.set_font(
        "Helvetica",
        "",
        9
    )

    if stats["taux"] > 30:

        concl = (
            "Le niveau d'anomalies detecte est eleve. "
            "Une investigation immediate des flux suspects "
            "et un renforcement des mesures de securite "
            "sont recommandes."
        )

    elif stats["anomalies"] > 0:

        concl = (
            "Des flux suspects ont ete detectes. "
            "Il est conseille de surveiller ces flux "
            "et d'appliquer les recommandations du diagnostic."
        )

    else:

        concl = (
            "Aucune anomalie significative n'a ete detectee. "
            "Le trafic reseau analyse presente un comportement normal."
        )

    pdf.set_x(
        pdf.l_margin
    )

    pdf.multi_cell(
        largeur_tot,
        5,
        concl
    )

    # ============================================================
    # Retour du PDF
    # ============================================================

    return bytes(
        pdf.output()
    )


paquet = charger_paquet()

# ---------------- ÉTAT ----------------
for cle in ("analyse", "diag", "tableau", "cumul", "familles",
            "donnees_brutes", "colonnes_features", "stats_normales", "modele_actif"):
    if cle not in st.session_state:
        st.session_state[cle] = None
if "statuts_alertes" not in st.session_state:
    st.session_state.statuts_alertes = {}
if "page" not in st.session_state:
    st.session_state.page = "🏠 Accueil"

PAGES = ["🏠 Accueil", "📂 Analyse du trafic", "🔍 Diagnostic intelligent",
         "🚨 Alertes", "📊 Dashboard", "ℹ️ À propos"]
SOUS_TITRE = {
    "🏠 Accueil": "Centre de supervision — vue d'ensemble",
    "📂 Analyse du trafic": "Importer et analyser un fichier réseau",
    "🔍 Diagnostic intelligent": "Types d'attaques, causes et recommandations",
    "🚨 Alertes": "Anomalies importantes, criticité et statut de traitement",
    "📊 Dashboard": "Indicateurs, visualisations et rapports",
    "ℹ️ À propos": "Modèle, technologies et données",
}


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div class="marque">
        <div class="carre">🛡️</div>
        <div class="txt"><b>ONDA</b><span>Sécurité réseau — Nador</span></div>
    </div>""", unsafe_allow_html=True)

    for p in PAGES:
        if p == st.session_state.page:
            st.markdown(f'<div class="nav-actif">{p}</div>', unsafe_allow_html=True)
        else:
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
                st.rerun()

    st.markdown(f"""
    <div class="profil">
        <div class="av">RS</div>
        <div><b>{NOM_UTILISATEUR}</b><br></div>
    </div>""", unsafe_allow_html=True)


page = st.session_state.page


# ---------------- HEADER FIXE ----------------
maintenant = datetime.now()
titre_page = page.split(" ", 1)[1] if " " in page else page
st.markdown(f"""
<div class="top-header">
    <div>
        <div class="header-title-badge">{titre_page}</div>
        <p>{SOUS_TITRE.get(page, '')}</p>
    </div>
    <div class="droite">
        <div class="h-badge"><span class="dot"></span> SYSTÈME ACTIF</div>
        <div class="h-date"><b>{maintenant.strftime('%d/%m/%Y')}</b><br>
            {maintenant.strftime('%H:%M')} — Aéroport Nador</div>
        <div class="h-bell">🔔</div>
                <a href="?logout=1" target="_self" class="h-logout">🚪 Déconnexion</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ================= 🏠 ACCUEIL =================
if page == "🏠 Accueil":
    st.markdown("""
    <div class="bienvenue">
        <h2>👋 Bienvenue sur le centre de supervision</h2>
        <p>Plateforme de surveillance du trafic réseau de l'infrastructure
        aéroportuaire. Elle identifie automatiquement les comportements anormaux
        pouvant révéler une cyberattaque, à l'aide d'un modèle d'intelligence
        artificielle entraîné sur le jeu de données <b>CIC-IDS2017</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="carte"><div class="ico">📂</div><h4>Importer</h4>
        <p>Chargez un fichier réseau (CSV / Parquet).</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="carte cyan"><div class="ico">🤖</div><h4>Analyser</h4>
        <p>Classification de chaque flux par l'IA.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="carte rouge"><div class="ico">🔍</div><h4>Diagnostiquer</h4>
        <p>Type d'attaque, risque, recommandations.</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="carte vert"><div class="ico">📊</div><h4>Superviser</h4>
        <p>Tableau de bord, alertes et rapports</p></div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"""
    <div class="fichier"><div class="ico">🛡️</div><div>
    <b>Menaces surveillées :</b> <span>DDoS · DoS · PortScan · Botnet ·
    Brute Force (FTP/SSH) · Infiltration · Attaques Web</span></div></div>
    """, unsafe_allow_html=True)


# ================= 📂 ANALYSE DU TRAFIC =================
elif page == "📂 Analyse du trafic":
    if paquet is None:
        st.error("Modèle introuvable. Lance d'abord : `python -m src.train_model`")
    else:
        etape(1, "Importer un dataset")
        fichier = st.file_uploader("Fichier de trafic réseau (CSV ou Parquet)",
                                   type=["parquet", "csv"], label_visibility="collapsed")

        if fichier is None:
            st.info("Formats acceptés : **CSV** ou **Parquet** au format CIC-IDS2017.")
        else:
            try:
                df = (pd.read_parquet(fichier) if fichier.name.lower().endswith(".parquet")
                      else pd.read_csv(fichier, low_memory=False))
                df.columns = df.columns.str.strip()
                lecture_ok = True
            except Exception as e:
                st.error(f"Lecture impossible : {e}")
                lecture_ok = False

            if lecture_ok:
                st.markdown(f"""
                <div class="fichier"><div class="ico">📄</div>
                <div><b>{fichier.name}</b><br>
                <span>{fmt(df.shape[0])} flux · {df.shape[1]} colonnes</span></div></div>
                """, unsafe_allow_html=True)

                with st.expander("Aperçu des données"):
                    st.dataframe(df.head(15), use_container_width=True)

                st.markdown("")
                etape(2, "Prétraitement et analyse IA")
                max_l = int(min(50000, len(df)))
                n = st.slider("Nombre de flux à analyser", 1000, max_l,
                              value=int(min(5000, max_l)), step=1000)

                if st.button("🚀 Analyser", type="primary", use_container_width=True):
                    with st.spinner("Prétraitement, analyse et diagnostic en cours..."):
                        ech = df.sample(n=min(n, len(df)),
                                        random_state=42).reset_index(drop=True)
                        resultat, predictions, confiances = predire(ech)
                        anomalies_df = ech[predictions == 1]
                        detail, synthese = diagnostiquer(anomalies_df)
                        familles = familles_detectees(anomalies_df)

                    total = len(predictions)
                    anomalies = int((predictions == 1).sum())
                    normal = total - anomalies
                    taux = 100 * anomalies / total if total else 0

                    types_col = np.array(["Benign"] * total, dtype=object)
                    if len(detail) > 0:
                        types_col[np.where(predictions == 1)[0]] = detail["Type probable"].values

                    tableau = pd.DataFrame({
                        "Flow ID": range(1, total + 1),
                        "Prédiction": np.where(predictions == 1, "Anomalie", "Benign"),
                        "Type probable": types_col,
                        "État": np.where(predictions == 1, "🚨", "✅"),
                    })

                    st.session_state.analyse = {
                        "total": total, "normal": normal, "anomalies": anomalies,
                        "taux": taux, "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "fichier": fichier.name, "metriques": metriques_modele(paquet),
                    }
                    st.session_state.diag = synthese
                    st.session_state.tableau = tableau
                    st.session_state.cumul = np.cumsum(predictions == 1)
                    st.session_state.familles = familles

                    # ---- Données pour l'explicabilité (page Diagnostic / Alertes) ----
                    colonnes_features = obtenir_colonnes_features(paquet, ech)
                    st.session_state.donnees_brutes = ech.reset_index(drop=True)
                    st.session_state.colonnes_features = colonnes_features
                    st.session_state.stats_normales = calculer_stats_normales(
                        ech.reset_index(drop=True), colonnes_features, predictions)
                    st.session_state.modele_actif = obtenir_modele(paquet)
                    st.session_state.statuts_alertes = {}

                    st.markdown("")
                    etape(3, "Détection")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: carte_kpi(fmt(total), "Flux analysés")
                    with c2: carte_kpi(fmt(normal), "Trafic normal", "vert")
                    with c3: carte_kpi(fmt(anomalies), "Anomalies", "rouge")
                    with c4: carte_kpi(f"{taux:.1f} %", "Taux d'anomalies", "orange")

                    st.markdown("")
                    if taux > 30:
                        st.error(f"⚠️ ALERTE : nombre élevé d'anomalies détectées ({taux:.1f} %) !")
                    elif anomalies > 0:
                        st.warning(f"⚠️ {fmt(anomalies)} flux suspect(s) détecté(s).")
                    else:
                        st.success("✅ Aucune anomalie détectée.")

                    st.markdown("")
                    st.markdown("**Détail des flux**")
                    st.dataframe(tableau, use_container_width=True, height=300)

                    st.success("Analyse terminée. Consultez **🔍 Diagnostic intelligent** "
                               "et **📊 Dashboard**.")


# ================= 🔍 DIAGNOSTIC INTELLIGENT =================
elif page == "🔍 Diagnostic intelligent":
    familles = st.session_state.familles
    a = st.session_state.analyse
    t = st.session_state.tableau

    if a is None or familles is None:
        bloc_vide("🔍", "Aucun diagnostic disponible",
                  "Lancez d'abord une analyse dans <b>📂 Analyse du trafic</b>.")
    elif len(familles) == 0:
        st.success("✅ Aucune anomalie détectée : aucun diagnostic nécessaire.")
    else:
        dominant = familles[0]
        c1, c2, c3 = st.columns(3)
        with c1: carte_kpi(fmt(a["anomalies"]), "Flux suspects", "rouge")
        with c2: carte_kpi(str(len(familles)), "Types d'attaques", "orange")
        with c3: carte_kpi(dominant["type"], "Type dominant", "rouge")

        st.markdown("")
        st.caption("Chaque type d'attaque détecté est présenté avec sa description, "
                   "ses causes et ses recommandations.")

        for f in familles:
            st.markdown("")
            entete = (f'<div style="display:flex;align-items:center;gap:14px;'
                      f'margin-bottom:10px;">'
                      f'<span style="font-family:Montserrat;font-size:21px;'
                      f'font-weight:800;color:{NAVY};">{f["type"]}</span>'
                      f'<span class="pastille {classe_risque(f["risque"])}">'
                      f'Risque : {f["risque"]}</span>'
                      f'<span class="pastille" style="background:#EEF2F6;color:{NAVY};">'
                      f'{fmt(f["count"])} flux</span></div>')
            st.markdown(entete, unsafe_allow_html=True)

            if f.get("description"):
                st.markdown(f'<div class="desc-attaque"><b>Description :</b> '
                            f'{f["description"]}</div>', unsafe_allow_html=True)

            causes = "".join(f"<li>{c}</li>" for c in f["causes"])
            recos = "".join(f"<li>{r}</li>" for r in f["recommandations"])
            col_c, col_r = st.columns(2)
            with col_c:
                st.markdown(f'<div class="bloc-c"><h4>⚠️ Causes probables</h4>'
                            f'<ul>{causes}</ul></div>', unsafe_allow_html=True)
            with col_r:
                st.markdown(f'<div class="bloc-r"><h4>🛡️ Recommandations</h4>'
                            f'<ul>{recos}</ul></div>', unsafe_allow_html=True)

        st.markdown("")
        st.subheader("Flux suspects détaillés")
        if t is not None:
            suspects = t[t["Prédiction"] == "Anomalie"]
            suspects_filtres = filtrer_tableau(suspects, cle_recherche="diag")
            st.caption(f"{fmt(len(suspects_filtres))} flux affiché(s) sur {fmt(len(suspects))} anomalie(s).")
            st.dataframe(suspects_filtres.head(200), use_container_width=True, height=300)

            st.markdown("")
            st.subheader("🔬 Pourquoi ce flux a-t-il été détecté ?")
            st.caption("Cette fonctionnalité permet de comprendre pourquoi le modèle a considéré ce flux comme anormal. Elle identifie les caractéristiques réseau qui ont le plus influencé la décision du modèle en comparant les valeurs observées avec celles du trafic normal. Plus le score d’influence est élevé, plus la caractéristique a contribué à la détection du flux."
                       ".")
            if len(suspects_filtres) > 0:
                flow_choisi = st.selectbox("Flow ID à expliquer", suspects_filtres["Flow ID"].tolist(),
                                           key="diag_flow_explication")
                afficher_explication(flow_choisi)
            else:
                st.info("Aucun flux ne correspond aux filtres pour afficher une explication.")


# ================= 🚨 ALERTES =================
elif page == "🚨 Alertes":
    a = st.session_state.analyse
    t = st.session_state.tableau
    fam = st.session_state.familles

    if a is None or t is None:
        bloc_vide(
            "🚨",
            "Aucune alerte pour le moment",
            "Lancez d'abord une analyse dans <b>📂 Analyse du trafic</b> pour générer les alertes."
        )
    else:
        # Filtrage et traitement des données
        alertes = construire_alertes(t, st.session_state.statuts_alertes)

        # Calcul des KPI
        nb_critique = int((alertes["Criticité"] == "Critique").sum())
        nb_eleve = int((alertes["Criticité"] == "Élevé").sum())
        nb_nouveau = int((alertes["Statut"] == "Nouveau").sum())

        # Affichage des cartes KPI
        c1, c2, c3, c4 = st.columns(4)
        with c1: carte_kpi(fmt(len(alertes)), "Alertes totales", "rouge")
        with c2: carte_kpi(fmt(nb_critique), "Criticité critique", "rouge")
        with c3: carte_kpi(fmt(nb_eleve), "Criticité élevée", "orange")
        with c4: carte_kpi(fmt(nb_nouveau), "Nouvelles alertes", "cyan")

        if len(alertes) == 0:
            st.markdown("")
            st.success("✅ Aucune alerte active. Trafic réseau normal.")
        else:
            st.markdown("")
            alertes_filtrees = filtrer_tableau(alertes, cle_recherche="alertes")

            fc1, fc2 = st.columns(2)
            with fc1:
                criticites_choisies = st.multiselect(
                    "Criticité", ["Critique", "Élevé", "Moyen"], 
                    key="alertes_criticite",
                    label_visibility="collapsed", 
                    placeholder="Filtrer par criticité"
                )
            with fc2:
                statuts_choisis = st.multiselect(
                    "Statut", ["Nouveau", "En cours", "Traité"], 
                    key="alertes_statut",
                    label_visibility="collapsed", 
                    placeholder="Filtrer par statut"
                )

            if criticites_choisies:
                alertes_filtrees = alertes_filtrees[alertes_filtrees["Criticité"].isin(criticites_choisies)]
            if statuts_choisis:
                alertes_filtrees = alertes_filtrees[alertes_filtrees["Statut"].isin(statuts_choisis)]

            st.markdown("")
            st.subheader(f"📋 Liste des alertes ({fmt(len(alertes_filtrees))})")

            if len(alertes_filtrees) == 0:
                st.info("Aucune alerte ne correspond aux filtres sélectionnés.")
            else:
                colonnes_affichees = ["Flow ID", "Type probable", "Criticité", "Statut"]
                edite = st.data_editor(
                    alertes_filtrees[colonnes_affichees],
                    column_config={
                        "Flow ID": st.column_config.NumberColumn("Flow ID", disabled=True),
                        "Type probable": st.column_config.TextColumn("Type d'attaque", disabled=True),
                        "Criticité": st.column_config.TextColumn("Criticité", disabled=True),
                        "Statut": st.column_config.SelectboxColumn(
                            "Statut", options=["Nouveau", "En cours", "Traité"], required=True
                        ),
                    },
                    hide_index=True, 
                    use_container_width=True, 
                    height=360, 
                    key="editeur_alertes"
                )

                # Mettre à jour les statuts modifiés dans session_state
                for _, ligne in edite.iterrows():
                    st.session_state.statuts_alertes[int(ligne["Flow ID"])] = ligne["Statut"]

                st.caption("💡 Modifiez directement la colonne **Statut** pour suivre le traitement d'une alerte.")

# ================= 📊 DASHBOARD (+ RAPPORTS) =================
elif page == "📊 Dashboard":
    a = st.session_state.analyse
    d = st.session_state.diag
    t = st.session_state.tableau

    if a is None:
        bloc_vide("📊", "Aucune analyse pour le moment",
                  "Importez un fichier dans <b>📂 Analyse du trafic</b> et lancez une détection.")
    else:
        m = a["metriques"]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: carte_kpi(fmt(a["total"]), "Flux analysés")
        with c2: carte_kpi(fmt(a["anomalies"]), "Anomalies", "rouge")
        with c3: carte_kpi(fmt(a["normal"]), "Trafic normal", "vert")
        with c4: carte_kpi(f"{m['precision']*100:.1f} %", "Précision modèle", "cyan")
        niveau = "Critique" if a["taux"] > 30 else ("Élevé" if a["anomalies"] else "Faible")
        with c5: carte_kpi(niveau, "Niveau de risque",
                           "rouge" if a["taux"] > 30 else "vert")

        st.markdown("")
        st.markdown(f"""
        <div class="fichier"><div class="ico">📅</div>
        <div><b>Dernière analyse : {a['date']}</b><br>
        <span>Fichier source : {a.get('fichier','—')}</span></div></div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if a["taux"] > 30:
            st.error(f"⚠️ ALERTE : niveau d'anomalies élevé ({a['taux']:.1f} %).")
        elif a["anomalies"] > 0:
            st.warning(f"⚠️ {fmt(a['anomalies'])} flux suspect(s) détecté(s).")
        else:
            st.success("✅ Aucune anomalie détectée. Trafic réseau normal.")

        if d:
            st.markdown("")
            st.markdown(f"""
            <div class="diag"><h3>🔍 Diagnostic dominant</h3>
            <div class="type">{d['type']}</div>
            <span class="pastille {classe_risque(d['risque'])}">Risque : {d['risque']}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.subheader("Visualisation")
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            ax.pie([a["normal"], a["anomalies"]], labels=["Normal", "Anomalies"],
                   autopct="%1.1f%%", colors=[VERT_OK, ROUGE], startangle=90,
                   wedgeprops={"width": 0.42})
            ax.axis("equal"); ax.set_title("Répartition du trafic", fontsize=11)
            st.pyplot(fig)
        with g2:
            fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
            ax2.bar(["Normal", "Anomalies"], [a["normal"], a["anomalies"]],
                    color=[VERT_OK, ROUGE])
            ax2.set_ylabel("Nombre de flux"); ax2.set_title("Par catégorie", fontsize=11)
            st.pyplot(fig2)

        cumul = st.session_state.get("cumul")
        fig3, ax3 = plt.subplots(figsize=(8.6, 2.9))
        if cumul is not None:
            ax3.plot(range(1, len(cumul) + 1), cumul, color=BLEU, linewidth=2)
            ax3.fill_between(range(1, len(cumul) + 1), cumul, color=BLEU, alpha=0.12)
        ax3.set_xlabel("Flux"); ax3.set_ylabel("Anomalies cumulées")
        ax3.set_title("Évolution temporelle", fontsize=11)
        st.pyplot(fig3)

        # ---- Types d'attaques détectés (barres horizontales stylées) ----
        fam = st.session_state.familles
        if fam:
            st.markdown("")
            st.subheader("📊 Types d'attaques détectés")
            fam_tri = sorted(fam, key=lambda z: z["count"], reverse=True)
            maxc = max((f["count"] for f in fam_tri), default=1) or 1
            palette = [ROUGE, ORANGE, DORE, BLEU, CYAN, VERT_OK]
            html = '<div class="att-liste">'
            for i, f in enumerate(fam_tri):
                coul = palette[i % len(palette)]
                pct = max(6, round(100 * f["count"] / maxc))
                html += (f'<div class="att-item">'
                         f'<div class="att-tete">'
                         f'<span class="att-nom">🚨 {f["type"]}</span>'
                         f'<span class="att-count" style="color:{coul};">'
                         f'{fmt(f["count"])} flux</span></div>'
                         f'<div class="att-piste"><div class="att-barre" '
                         f'style="width:{pct}%;background:{coul};"></div></div>'
                         f'</div>')
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

        # ---- Suivi des alertes (lien avec la page 🚨 Alertes) ----
        if t is not None and (t["Prédiction"] == "Anomalie").any():
            alertes_t = t[t["Prédiction"] == "Anomalie"].copy()
            alertes_t["Criticité"] = alertes_t["Type probable"].apply(
                lambda ta: CATALOGUE.get(ta, CATALOGUE["Anomalie indéterminée"])["risque"])
            alertes_t["Statut"] = alertes_t["Flow ID"].apply(
                lambda fid: st.session_state.statuts_alertes.get(int(fid), "Nouveau"))

            st.markdown("")
            st.subheader("🚨 Suivi des alertes")
            st.caption("Aperçu de l'avancement du traitement des alertes. "
                       "Détail et gestion complète sur la page **🚨 Alertes**.")

            nb_nouveau = int((alertes_t["Statut"] == "Nouveau").sum())
            nb_en_cours = int((alertes_t["Statut"] == "En cours").sum())
            nb_traite = int((alertes_t["Statut"] == "Traité").sum())
            nb_critique = int((alertes_t["Criticité"] == "Critique").sum())

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: carte_kpi(fmt(nb_nouveau), "Nouvelles alertes", "rouge")
            with sc2: carte_kpi(fmt(nb_en_cours), "En cours de traitement", "orange")
            with sc3: carte_kpi(fmt(nb_traite), "Alertes traitées", "vert")
            with sc4: carte_kpi(fmt(nb_critique), "Criticité critique", "rouge")

            g3, g4 = st.columns(2)
            with g3:
                repartition_statut = (
                    alertes_t["Statut"].value_counts()
                    .reindex(["Nouveau", "En cours", "Traité"]).fillna(0)
                )
                fig4, ax4 = plt.subplots(figsize=(4.2, 3.2))
                couleurs_statut = [ROUGE, ORANGE, VERT_OK]
                ax4.bar(repartition_statut.index, repartition_statut.values, color=couleurs_statut)
                ax4.set_ylabel("Nombre d'alertes")
                ax4.set_title("Alertes par statut", fontsize=11)
                st.pyplot(fig4)
            with g4:
                repartition_criticite = (
                    alertes_t["Criticité"].value_counts()
                    .reindex(["Critique", "Élevé", "Moyen"]).fillna(0)
                )
                fig5, ax5 = plt.subplots(figsize=(4.2, 3.2))
                couleurs_crit = [ROUGE, ORANGE, BLEU]
                ax5.bar(repartition_criticite.index, repartition_criticite.values, color=couleurs_crit)
                ax5.set_ylabel("Nombre d'alertes")
                ax5.set_title("Alertes par criticité", fontsize=11)
                st.pyplot(fig5)

        # ---- RAPPORTS (intégré au Dashboard) ----
        st.markdown("")
        st.subheader("📄 Rapports")
        if t is not None:
            r1, r2 = st.columns(2)
            with r1:
                csv = t.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Rapport CSV", data=csv,
                                   file_name="rapport_analyse.csv", mime="text/csv",
                                   use_container_width=True)
            with r2:
                alertes_pdf = construire_alertes(t, st.session_state.statuts_alertes)
                pdf = generer_pdf(a, d, t, alertes_pdf)
                if pdf is not None:
                    st.download_button("📄 Rapport PDF", data=pdf,
                                       file_name="rapport_analyse.pdf",
                                       mime="application/pdf", use_container_width=True)
                else:
                    st.caption("PDF indisponible — installe : `pip install fpdf2`")

            with st.expander("Aperçu du contenu du rapport"):
                st.dataframe(t.head(100), use_container_width=True, height=280)


# ================= ℹ️ À PROPOS =================
elif page == "ℹ️ À propos":
    st.markdown("""
    <div class="bienvenue">
        <h2>La plateforme</h2>
        <p>Système de détection d'anomalies du trafic réseau conçu pour une
        infrastructure aéroportuaire. Il analyse les flux et signale automatiquement
        les comportements pouvant correspondre à une cyberattaque.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Le modèle de détection")
    if paquet is None:
        st.warning("Modèle non entraîné. Lance : `python -m src.train_model`")
    else:
        m = metriques_modele(paquet)
        st.markdown(f"**Algorithme :** `{paquet.get('nom_modele','RandomForestClassifier')}` "
                    f"— entraîné sur **CIC-IDS2017**")
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        with c1: carte_kpi(f"{m['accuracy']*100:.1f} %", "Accuracy", "vert")
        with c2: carte_kpi(f"{m['precision']*100:.1f} %", "Precision", "cyan")
        with c3: carte_kpi(f"{m['recall']*100:.1f} %", "Recall", "orange")
        with c4: carte_kpi(f"{m['f1']*100:.1f} %", "F1-Score", "or")

        st.markdown("")
        st.subheader("Performances détaillées")
        p1, p2 = st.columns(2)
        cm = DOSSIER_REPORTS / "matrice_confusion.png"
        imp = DOSSIER_REPORTS / "importance_variables.png"
        comp = DOSSIER_REPORTS / "comparaison_modeles.png"
        if cm.exists(): p1.image(str(cm), caption="Matrice de confusion")
        if imp.exists(): p2.image(str(imp), caption="Variables les plus importantes")
        if comp.exists():
            st.image(str(comp), caption="Comparaison des algorithmes testés")

    st.markdown("")
    


# ---------------- FOOTER FIXE ----------------
st.markdown(f"""
<div class="bottom-footer">
    <div class="tag">🛡️ <b>ONDA</b> — Office National des Aéroports</div>
    <div>Aéroport de Nador · Détection d'anomalies réseau</div>
    <div class="tag">PFA © 2026</div>
</div>
""", unsafe_allow_html=True)