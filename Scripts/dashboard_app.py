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

# Dans notre cas les données sont sur google drive :
file_id = "192aA6koh_zobb6EjsbAcUERUsDZN-Prw"
url = f"https://drive.google.com/uc?id={file_id}"
output = "data.csv"

# Telechargement
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
    
# Chargement
try:
    data = pd.read_csv(output)
    
except Exception as e:
    st.error(f"Erreur de chargement du fichier: {e}")
    st.stop()

features = joblib.load("./Models/features.pkl")

# Sélection d'un ID client (et remplacement des ID vides)
client_id = st.selectbox("Sélectionnez un client :", data["SK_ID_CURR"].unique())
client_data = (

    data[data["SK_ID_CURR"] == client_id]
    .drop(columns=["TARGET", "SK_ID_CURR"])
    #.drop(columns=["TARGET", "SK_ID_CURR"])

    .fillna(-999)
    .infer_objects(copy=False))

st.subheader("Informations Client")

client_display = client_data.T.astype(str)
st.write(client_display)

# Appel à l'api de prédiction

if st.button("Évaluer l’éligibilité"):

    payload = client_data.iloc[0].to_dict()
    response = requests.post("http://localhost:8000/predict", json=payload, headers=headers)
    
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

