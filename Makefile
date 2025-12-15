# Makefile pour project-churn

# Variables
PYTHON=python
RAW_DIR=data/raw
PROCESSED_DIR=data/processed
PRED_DIR=data/predictions
STREAMLIT_APP=app/app_streamlit.py

# 1. ETL
etl:
	@echo "📦 Lancement de l'ETL..."
	$(PYTHON) -c "from src.etl import run_etl; run_etl(RAW_DIR, PROCESSED_DIR)"
	@echo "✅ ETL terminé. Fichier traité dans $(PROCESSED_DIR)/processed.csv"

# 2. Train
train:
	@echo "🏋️ Lancement du training..."
	$(PYTHON) -c "from src.models.train import train; train('$(PROCESSED_DIR)/processed.csv')"
	@echo "✅ Training terminé. Modèles sauvegardés dans models/"

# 3. Predict
predict:
	@echo "🤖 Génération des prédictions..."
	$(PYTHON) -c "from src.models.predict import predict; predict('$(PROCESSED_DIR)/processed.csv', '$(PRED_DIR)/predictions_output.csv')"
	@echo "✅ Prédictions sauvegardées dans $(PRED_DIR)/predictions_output.csv"

# 4. Streamlit
streamlit:
	@echo "🌐 Lancement de Streamlit..."
	streamlit run $(STREAMLIT_APP)

# 5. Tout (ETL + Train + Predict + Streamlit)
all: etl train predict streamlit

