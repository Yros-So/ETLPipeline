import pandas as pd
from pathlib import Path

def export_to_powerbi(input_csv: str, output_dir: str = "data/powerbi"):

    df = pd.read_csv(input_csv)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    file = output / "powerbi_dataset.csv"
    df.to_csv(file, index=False)

    print(f"Dataset exporté pour PowerBI → {file}")

    return file
