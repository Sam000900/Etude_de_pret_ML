# Prêt d'éligibilité - Néo Banque

## Description
Ce projet implémente un système d'évaluation de l'éligibilité au prêt pour une banque en ligne (Néo Banque). Le modèle prédit la probabilité qu'un client soit éligible à un prêt, en se basant sur un jeu de données. Ce projet comprend trois principaux composants :

1. **Notebook d'entraînement** : Prépare les données et entraîne un modèle de machine learning pour prédire l'éligibilité.
2. **API Flask** : Fournit une interface RESTful permettant d'obtenir des prédictions sur l'éligibilité d'un client.
3. **Dashboard Streamlit** : Interface utilisateur permettant de sélectionner un client et d'afficher sa probabilité d'éligibilité au prêt.

## Prérequis
Avant de démarrer, assurez-vous d'avoir les dépendances suivantes installées :

```bash
pip install -r requirements.txt
```

**Les dépendances requises sont :**

*Flask :* pour créer l'API RESTful.

*Streamlit :* pour le tableau de bord interactif.

*Scikit-learn :* pour le prétraitement des données et les modèles de machine learning.

*XGBoost :* pour le modèle de classification.

*Imbalanced-learn :* pour les techniques de gestion du déséquilibre des classes.

*Joblib :* pour la sauvegarde et le chargement des modèles.

## Architecture du projet

### I - Notebook d'entraînement

Le notebook d'entraînement implémente plusieurs étapes de préparation des données, telles que :

- Chargement et nettoyage des données : Le dataset est d'abord nettoyé, les colonnes inutiles sont supprimées, et les valeurs manquantes sont traitées.

- Prétraitement : Un encodage One-Hot est appliqué pour convertir les données catégorielles en valeurs numériques. Les colonnes corrélées sont également supprimées.

- Gestion du déséquilibre des classes : La méthode d'undersampling est utilisée pour équilibrer les classes.

- Entraînement des modèles : Trois modèles sont entraînés et comparés : XGBoost, RandomForest et LogisticRegression. La recherche d'hyperparamètres est effectuée avec GridSearchCV.

- Évaluation des performances : Les performances des modèles sont évaluées à l'aide de la métrique AUC-ROC.

### II - API Flask

L'API Flask permet d'exposer le modèle sous forme de service web pour effectuer des prédictions d'éligibilité. Elle attend une requête POST avec des données client et renvoie un score d'éligibilité entre 0 et 1. Ce score est basé sur les prédictions du modèle.

**Points clés :**

- Sécurisation avec clé API : La clé API est nécessaire pour faire des requêtes vers l'API, conformément aux exigences RGPD.

- URL : /predict

- Méthode : POST

- Corps de la requête : Les données client doivent être envoyées en format JSON.

- Réponse : Le score d'éligibilité est renvoyé dans une réponse JSON.

### III - Dashboard Streamlit

Le dashboard Streamlit permet d'interagir avec l'API et d'afficher les résultats de manière visuelle. L'utilisateur peut sélectionner un client à partir d'un ID, voir ses informations et obtenir un score d'éligibilité.

**Fonctionnalités :**

- Sélection d'un client : L'utilisateur choisit un client parmi ceux présents dans le dataset.

- Affichage des informations client : Les informations de base du client sont affichées.

- Affichage du score d'éligibilité : Lorsque l'utilisateur clique sur "Évaluer l’éligibilité", le score est affiché et une étiquette ("Client à risque", "Client à étudier", etc.) est assignée en fonction du score.

# Utilisation

1. Entraîner le modèle
Pour entraîner le modèle et sauvegarder les artefacts nécessaires à l'API et au dashboard, exécutez le notebook d'entraînement.
Le temps de calcul de l'entrainement des modèle peux excéder les 40 minutes.

2. Démarrer l'API Flask 
```bash
python Scripts/prediction_api.py
```
Une fois le modèle et les artefacts sauvegardés, démarrez l'API Flask.
L'API sera accessible sur http://localhost:8000.

3. Démarrer le dashboard Streamlit
Lancez le dashboard avec la commande suivante :

```bash
streamlit run Scripts/dashboard_app.py
```

Accédez à l'interface web du dashboard via http://localhost:8501 pour interagir avec le modèle et obtenir des prédictions.

# Structure des fichiers
Voici la structure des fichiers du projet :

Root
├── Data
│   └── application_train.csv      # Dataset
├── Models
│   ├── model.pkl                  # Modèle entraîné
│   ├── scaler.pkl                 # Scaler pour normalisation
│   └── features.pkl               # Liste des caractéristiques utilisées
├── app.py                         # API Flask
├── dashboard.py                   # Dashboard Streamlit
├── requirements.txt               # Dépendances du projet
└── README.md                      # Documentation du projet