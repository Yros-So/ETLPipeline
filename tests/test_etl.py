import pandas as pd
from src.etl import basic_clean

import sys
from pathlib import Path

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

def test_basic_clean():
    # Données avec doublons et espaces
    df = pd.DataFrame({
        'num': [1, 1, 2],
        'cat': ['x ', ' x', 'y'],
        'val': [None, 2, 3]
    })

    out = basic_clean(df)

    # Vérifie que les espaces sont bien supprimés
    assert out['cat'].iloc[0] == 'x'
    assert out['cat'].iloc[1] == 'y'

    # Vérifie suppression des doublons (num=1 doit apparaître une seule fois)
    assert out['num'].duplicated().sum() == 0

    # Vérifie que les valeurs manquantes sont intactes si pas remplies
    assert pd.isna(out['val']).sum() <= 1
