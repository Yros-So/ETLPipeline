from dotenv import load_dotenv
import os

import sys
from pathlib import Path

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Charger les variables d'environnement depuis .env
load_dotenv()

# Définir les chemins de base et des dossiers de travail
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR', BASE_DIR / 'data' / 'raw'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', BASE_DIR / 'models'))
PREDICTIONS_DIR = Path(os.getenv('PREDICTIONS_DIR', BASE_DIR / 'predictions'))

# Créer les dossiers si nécessaire
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True) 

# Configuration de la base de données 
DB = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'churn_user'),
    'password': os.getenv('DB_PASSWORD', 'Jtm123mama'),
    'dbname': os.getenv('DB_NAME', 'churn_db')
}