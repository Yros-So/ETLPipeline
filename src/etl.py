import pandas as pd
from pathlib import Path
from src.features import build_feature_matrix

def run_etl(raw_dir: Path, processed_dir: Path = Path("data/processed")):

    processed_dir.mkdir(parents=True, exist_ok=True)
    files = list(raw_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {raw_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df_processed = build_feature_matrix(df)

    output = processed_dir / "processed.csv"
    df_processed.to_csv(output, index=False)

    return output
