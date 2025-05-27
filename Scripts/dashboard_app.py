import io
import os
import gdown
import joblib
import requests
import pandas as pd
import streamlit as st

st.title("Dashboard d'éligibilité au prêt - Néo Banque")
headers = {"ML-api-key": "super-secret-API-key"}



# Chargement des données (en tant normal si elle se trouve sur github
# data = pd.read_csv("../Data/application_test.csv") # si ca ne fonctionne pas : mettre ./Data/application_train.csv



# test 
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV et met en cache tant que le fichier n'a pas changé."""
    return pd.read_csv(path)

# ── 2️⃣ Téléchargement du fichier depuis Google Drive ────────────
file_id = "1Gi6chtMWmaHu80R5gu83_MUKeDp2BUek"
output  = "data.csv"

if not os.path.exists(output):
    try:
        gdown.download(
            id=file_id,         # on passe l'ID plutôt que l'URL complète
            output=output,
            quiet=False,
            fuzzy=True,
            use_cookies=False
        )
    except gdown.exceptions.FileURLRetrievalError as e:
        st.error(
            "Impossible de récupérer le fichier sur Google Drive.\n"
            "• Vérifiez les droits de partage\n"
            "• Attendez un peu si le quota est dépassé\n"
            "• Ou hébergez le CSV ailleurs."
        )
        st.stop()

# ── 3️⃣ Chargement en cache ──────────────────────────────────────
try:
    data = load_data(output)
except Exception as e:
    st.error(f"Erreur de chargement du fichier : {e}")
    st.stop()





features = joblib.load("./Models/features.pkl")

# Bandeau Latéral Gauche : Filtres 
with st.sidebar:
    st.header("Filtres de recherche")

    # Âge 
    if "AGE" not in data.columns:
        data["AGE"] = (-data["DAYS_BIRTH"] // 365).astype(int)

    age_min, age_max = int(data["AGE"].min()), int(data["AGE"].max())
    age_range = st.slider(
        "Âge",
        min_value=age_min, max_value=age_max,
        value=(age_min, age_max), step=1,
        key="age_slider"
    )

    # Revenu total 
    income_min, income_max = map(int, (data["AMT_INCOME_TOTAL"].min(),
                                       data["AMT_INCOME_TOTAL"].max()))
    income_range = st.slider(
        "Revenu total",
        min_value=income_min, max_value=income_max,
        value=(income_min, income_max), step=1000,
        format="%d",
        key="income_slider"
    )

    # Sexe 
    sex_selected = st.multiselect(
        "Sexe",
        options=["M", "F"],
        default=["M", "F"],
        key="sex_multiselect"
    )

    # Type de revenu 
    income_type_options = sorted(data["NAME_INCOME_TYPE"].unique())
    income_type_selected = st.multiselect(
        "Type de revenu",
        options=income_type_options,
        default=income_type_options,
        key="income_type_multiselect"
    )

    st.divider()  # petite ligne de séparation

    # Filtrage du dataframe 
    filtered_data = data[
        (data["AGE"].between(*age_range)) &
        (data["AMT_INCOME_TOTAL"].between(*income_range)) &
        (data["CODE_GENDER"].isin(sex_selected)) &
        (data["NAME_INCOME_TYPE"].isin(income_type_selected))
    ]

    st.caption(f"{len(filtered_data)} client(s) correspondant(s) aux filtres")

    # Sélecteur d'ID client 
    client_id = st.selectbox(
        "Sélectionnez un client :",
        filtered_data["SK_ID_CURR"].unique(),
        key="client_select"
    )



# Sélection d'un ID client (et remplacement des ID vides)
client_id = st.selectbox("Sélectionnez un client :", data["SK_ID_CURR"].unique())
client_data = (

    data[data["SK_ID_CURR"] == client_id]
    .drop(columns=["SK_ID_CURR"])
    #.drop(columns=["TARGET", "SK_ID_CURR"])

    .fillna(-999)
    .infer_objects(copy=False))

st.subheader("Informations Client")

client_display = client_data.T.astype(str)
st.write(client_display)

# Appel à l'api de prédiction

st.markdown(
    """
### Interprétation du Score d’éligibilité
Le score d’éligibilité au prêt est calculé en fonction de multiples facteurs financiers et personnels du client.

- Score inférieur à 0.3 : Le client est considéré comme à risque.  
- Score entre 0.3 et 0.6 : Le client doit être étudié plus en détail.  
- Score entre 0.6 et 0.85 : Le client est probablement éligible pour un prêt.  
- Score supérieur à 0.85 : Le client est éligible pour un prêt avec une faible probabilité de risque.  
    """
)

if st.button("Évaluer l’éligibilité"):

    payload = client_data.iloc[0].to_dict()
    response = requests.post("https://etude-de-pret-ml.onrender.com/predict", json=payload, headers=headers)
    
    if response.status_code == 200:

        score = response.json()["score"]
        st.metric("Score d’éligibilité (plus bas = risque)", score)
        
        # Calcul simple du score client

        if score < 0.3:
            st.error("Client à risque")
        elif score < 0.6:
            st.warning("Client à étudier")
        elif score < 0.85:
            st.success("Client probablement éligible")
        else:
            st.success("Client éligible")

    else:
        st.error("Erreur API")

