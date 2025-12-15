from sqlalchemy import create_engine
from .config import DB

import sys
from pathlib import Path

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))


def get_engine(debug=False):
    """Crée et retourne un moteur SQLAlchemy pour la base de données."""
    uri = f"postgresql+psycopg2://{DB['user']}:{DB['password']}@{DB['host']}:{DB['port']}/{DB['dbname']}"
    
    if debug:
        print("Using DB URI:", uri)
    return create_engine(uri, pool_pre_ping=True)


def write_predictions(df, table='predictions'):
    """Écrit un DataFrame dans la table spécifiée de la base de données."""
    engine = get_engine() 
    df.to_sql(table, engine, if_exists='append', index=False) 


def read_data(query):
    """Lit les données de la base de données en fonction de la requête SQL fournie."""
    engine = get_engine()
    with engine.connect() as connection:
        result = connection.execute(query)
        return result.fetchall() 

    
def initialize_db():
    """Initialise la base de données en créant les tables nécessaires."""
    engine = get_engine()
    with engine.connect() as connection:
        with open('infra/init_sql.sql', 'r') as file:
            init_sql = file.read()
            connection.execute(init_sql)


if __name__ == "__main__":
    initialize_db() 
    # Initialiser la base de données lors de l'exécution directe
    
