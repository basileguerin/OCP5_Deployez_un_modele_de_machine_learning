import streamlit as st
import requests

# Config
st.set_page_config(page_title="HRPredict", layout="wide")
st.title("HRPredict")
st.caption("Enter employee information to predict resignation risk.")

API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"

# Helpers
def one_hot(selected: str, options: list[str], prefix: str) -> dict:
    """Return one-hot dict for features like prefix_value."""
    d = {f"{prefix}{opt}": 0.0 for opt in options}
    d[f"{prefix}{selected}"] = 1.0
    return d

# One-hot options
POSTES = [
    "Assistant de Direction",
    "Cadre Commercial",
    "Consultant",
    "Directeur Technique",
    "Manager",
    "Représentant Commercial",
    "Ressources Humaines",
    "Senior Manager",
    "Tech Lead",
]

DEPARTEMENTS = [
    "Commercial",
    "Consulting",
    "Ressources Humaines",
]

DOMAINES = [
    "Autre",
    "Entrepreunariat",
    "Infra & Cloud",
    "Marketing",
    "Ressources Humaines",
    "Transformation Digitale",
]

STATUTS = [
    "Célibataire",
    "Divorcé(e)",
    "Marié(e)",
]

# UI
with st.form("formulaire_prediction"):
    col1, col2, col3 = st.columns(3)

    # -------- Colonne 1 : Informations générales --------
    with col1:
        st.subheader("Informations générales")

        age = st.slider("Âge", 18, 80, 30)

        genre_label = st.selectbox("Genre", ["Femme", "Homme"])
        genre = {"Femme": 0.0, "Homme": 1.0}[genre_label]

        revenu_mensuel = st.slider("Revenu mensuel (€)", 1000, 30000, 3000, step=100)

        nombre_experiences_precedentes = st.slider("Nombre d'expériences précédentes", 0, 15, 2)
        annee_experience_totale = st.slider("Années d'expérience totale", 0, 50, 5)
        annees_dans_l_entreprise = st.slider("Ancienneté dans l'entreprise", 0, 50, 3)
        annees_dans_le_poste_actuel = st.slider("Ancienneté dans le poste actuel", 0, 50, 2)

    # -------- Colonne 2 : Satisfaction et performance --------
    with col2:
        st.subheader("Satisfaction et évaluation")

        satisfaction_employee_environnement = st.slider("Satisfaction environnement", 1, 4, 3)
        satisfaction_employee_nature_travail = st.slider("Satisfaction nature du travail", 1, 4, 3)
        satisfaction_employee_equipe = st.slider("Satisfaction équipe", 1, 4, 3)
        satisfaction_employee_equilibre_pro_perso = st.slider("Équilibre vie pro/perso", 1, 4, 3)

        note_evaluation_precedente = st.slider("Évaluation précédente", 1, 4, 3)
        note_evaluation_actuelle = st.slider("Évaluation actuelle", 1, 4, 3)

        niveau_hierarchique_poste = st.slider("Niveau hiérarchique", 1, 5, 2)

        heure_supp_label = st.selectbox("Heures supplémentaires", ["Non", "Oui"])
        heure_supplementaires = {"Non": 0.0, "Oui": 1.0}[heure_supp_label]

        nombre_participation_pee = st.slider("Participation PEE", 0, 3, 0)
        nb_formations_suivies = st.slider("Formations suivies", 0, 10, 0)

    # -------- Colonne 3 : Situation professionnelle --------
    with col3:
        st.subheader("Situation professionnelle")

        distance_domicile_travail = st.slider("Distance domicile-travail (km)", 0, 50, 10)
        niveau_education = st.slider("Niveau d'éducation", 1, 5, 3)

        frequence_label = st.selectbox(
            "Fréquence des déplacements",
            ["Aucun", "Occasionnel", "Frequent"]
        )
        frequence_deplacement = {
            "Aucun": 0.0,
            "Occasionnel": 1.0,
            "Frequent": 2.0
        }[frequence_label]

        annees_depuis_la_derniere_promotion = st.slider("Années depuis la dernière promotion", 0, 20, 2)
        annes_sous_responsable_actuel = st.slider("Années avec le manager actuel", 0, 20, 2)

        augmentation_label = st.selectbox(
            "Augmentation précédente",
            ["11-15 %", "16-20 %", "21-25 %"]
        )
        augmentation_salaire_precedente_bin = {
            "11-15 %": 0.0,
            "16-20 %": 1.0,
            "21-25 %": 2.0
        }[augmentation_label]

        st.markdown("### Catégories")
        poste = st.selectbox("Poste", POSTES)
        departement = st.selectbox("Département", DEPARTEMENTS)
        domaine = st.selectbox("Domaine d'étude", DOMAINES)
        statut = st.selectbox("Statut marital", STATUTS)

    submitted = st.form_submit_button("Predict")

# Build payload + call API
if submitted:
    # Base numeric/binary features
    features = {
        "age": float(age),
        "genre": float(genre),
        "revenu_mensuel": float(revenu_mensuel),
        "nombre_experiences_precedentes": float(nombre_experiences_precedentes),
        "annee_experience_totale": float(annee_experience_totale),
        "annees_dans_l_entreprise": float(annees_dans_l_entreprise),
        "annees_dans_le_poste_actuel": float(annees_dans_le_poste_actuel),
        "satisfaction_employee_environnement": float(satisfaction_employee_environnement),
        "note_evaluation_precedente": float(note_evaluation_precedente),
        "niveau_hierarchique_poste": float(niveau_hierarchique_poste),
        "satisfaction_employee_nature_travail": float(satisfaction_employee_nature_travail),
        "satisfaction_employee_equipe": float(satisfaction_employee_equipe),
        "satisfaction_employee_equilibre_pro_perso": float(satisfaction_employee_equilibre_pro_perso),
        "note_evaluation_actuelle": float(note_evaluation_actuelle),
        "heure_supplementaires": float(heure_supplementaires),
        "nombre_participation_pee": float(nombre_participation_pee),
        "nb_formations_suivies": float(nb_formations_suivies),
        "distance_domicile_travail": float(distance_domicile_travail),
        "niveau_education": float(niveau_education),
        "frequence_deplacement": float(frequence_deplacement),
        "annees_depuis_la_derniere_promotion": float(annees_depuis_la_derniere_promotion),
        "annes_sous_responsable_actuel": float(annes_sous_responsable_actuel),
        "augmentation_salaire_precedente_bin": float(augmentation_salaire_precedente_bin),
    }

    # One-hot groups
    features.update(one_hot(poste, POSTES, "poste_"))
    features.update(one_hot(departement, DEPARTEMENTS, "departement_"))
    features.update(one_hot(domaine, DOMAINES, "domaine_etude_"))
    features.update(one_hot(statut, STATUTS, "statut_marital_"))
    
    meta = requests.get(f"{API_BASE}/metadata").json()
    expected = set(meta["features_order"])
    sent = set(features.keys())

    missing = expected - sent
    extra = sent - expected

    if missing or extra:
        st.error("Payload invalide vs modèle")
        st.write("Missing:", list(missing)[:10])
        st.write("Extra:", list(extra)[:10])
        st.stop()
    # Call API
    try:
        r = requests.post(PREDICT_URL, json={"features": features}, timeout=20)
    except requests.RequestException as e:
        st.error("API unreachable.")
        st.code(str(e))
        st.stop()

    if r.status_code != 200:
        st.error(f"Prediction failed ({r.status_code})")
        st.code(r.text)
        st.stop()

    res = r.json()
    st.subheader("Résultat")

    prob = float(res["probability"])
    pred = int(res["prediction"])

    st.metric("Probabilité de démission", f"{prob:.2%}")

    if pred == 1:
        st.error("Risque élevé de démission")
    else:
        st.success("Risque faible de démission")

    if "request_id" in res:
        st.write("ID de la requête (enregistré en base) :")
        st.code(res["request_id"])
