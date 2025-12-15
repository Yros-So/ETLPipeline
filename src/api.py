import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
import pandas as pd

from src.predict import load_model_assets, predict

app = FastAPI(title="Churn API", version="1.0")

model, scaler, features = load_model_assets()

@app.post("/predict")
def predict_customer(data: list[dict]):
    df = pd.DataFrame(data)
    df_pred = predict(df, model, scaler, features)
    
    return df_pred.to_dict(orient="records")