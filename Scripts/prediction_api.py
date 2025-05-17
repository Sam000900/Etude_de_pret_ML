import os
import joblib
import numpy as np
import pandas as pd

from waitress import serve
from dotenv import load_dotenv
from flask import Flask, request, jsonify

app = Flask(__name__)

load_dotenv()
API_KEY = os.getenv("API_KEY")

model = joblib.load("./Models/model.pkl")
scaler = joblib.load("./Models/scaler.pkl")
features = joblib.load("./Models/features.pkl")

@app.route('/predict', methods=['POST'])

def predict():

    key = request.headers.get("ML-api-key")

    # Comme demandé par les lois RGPD : nous protégeons notre logiciel
    # L'utilisation d'une clé d'API permet cela

    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:

        input_data = request.get_json()

        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df).reindex(columns=features, fill_value=0)
    
        scaled = scaler.transform(df)
        score = model.predict_proba(scaled)[0][1]  
    
        return jsonify({"score": round(float(score), 3)})
    
    except Exception as e:

        print("Erreur lors de la prédiction :", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8000)