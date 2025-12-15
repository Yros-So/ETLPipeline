🧠 Projet Churn – Pipeline ETL + ML + Streamlit + Power BI + Docker
🔥 Description

Ce projet implémente un pipeline complet de Data Science / Machine Learning destiné à prédire le churn client.

Il inclut :

ETL Pipeline (fusion, nettoyage, feature engineering)

Machine Learning (Logistic Regression, Random Forest, XGBoost)

Sélection automatique du meilleur modèle

Sauvegarde du modèle + scaler + features.json

Dashboard Streamlit interactif

Export Power BI

Orchestration Prefect

Architecture Production-Ready

Exécution Docker / Docker Compose

Le tout est 100% automatisé, reproductible et prêt pour la mise en production.

project-churn/
│
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
│
├─ data/
│   ├─ raw/            # Données brutes
│   ├─ processed/      # Données ETL
│   └─ predictions/    # Sorties de prédictions
│
├─ models/
│   ├─ best_model.joblib
│   ├─ scaler.joblib
│   └─ features.json
│
├─ src/
│   ├─ app_streamlit.py
│   ├─ features.py
│   ├─ etl.py
│   ├─ pipeline.py
│   ├─ export_powerbi.py
│   │
│   ├─ visualisation.py
│   │
│   ├─ models/
│   │   ├─ utils.py
│   │   ├─ train.py
│   │   └─ predict.py
│   │
│   └─ db.py   (optionnel)
│
└─ tests/
    ├─ test_etl.py
    ├─ test_train.py
    ├─ test_predict.py
    └─ test_features.py