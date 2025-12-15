import pandas as pd
from src.predict import predict_dataframe

import sys
from pathlib import Path

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent)) 


def test_predict_dataframe():
    df = pd.DataFrame({
        'customer_id': ['c1', 'c2'],
        'birth_date': ['2000-01-01', '1995-05-05'],
        'gender': ['M', 'F']
    })
    res = predict_dataframe(df)
    assert 'prediction' in res.columns
    assert 'prob' in res.columns
    assert res['prob'].between(0,1).all()

