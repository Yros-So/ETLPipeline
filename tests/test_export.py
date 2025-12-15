import pandas as pd
from src.export_powerbi import export_for_powerbi
from pathlib import Path

import sys
from pathlib import Path

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))


def test_export_for_powerbi(tmp_path):
    df = pd.DataFrame({'a':[1,2]})
    out_file = export_for_powerbi(df, fname='test.csv', out_dir=tmp_path)
    assert Path(out_file).exists()

