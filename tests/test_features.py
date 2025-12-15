import pandas as pd
from src import BasicFeatureEngineer
import pytest
from src import BasicFeatureEngineer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))


def test_build_feature_matrix():
    df = pd.DataFrame({
        'birth_date': ['2000-01-01', '1990-06-15', None],
        'gender': ['M', 'F', None],
        'value': [10, 20, None]
    })
    X = BasicFeatureEngineer.build_feature_matrix(df)
    assert 'age' in X.columns
    assert X.isnull().sum().sum() == 0
    assert all(X.dtypes == float) or all(X.dtypes == int)

