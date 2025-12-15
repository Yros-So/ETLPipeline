import pandas as pd
from typing import Tuple, List

def build_feature_matrix(df: pd.DataFrame, return_features: bool = False):

    df = df.copy()

    # Nettoyage string
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace(
            {'': None, 'nan': None, 'None': None, 'none': None}
        )

    # Bool → int
    for c in df.select_dtypes(include=['bool']).columns:
        df[c] = df[c].astype(int)

    # Total Services
    services_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    services_cols = [c for c in services_cols if c in df.columns]

    if services_cols:
        df["TotalServices"] = df.apply(
            lambda row: sum(str(row.get(c, "no")).lower().startswith("yes") for c in services_cols),
            axis=1
        )
        
    # One-hot encoding
    small_cat_cols = [c for c in obj_cols if df[c].nunique() <= 10]
    df = pd.get_dummies(df, columns=small_cat_cols, drop_first=True)

    # Imputation
    for c in df.select_dtypes(include=['number']).columns:
        df[c] = df[c].fillna(df[c].median())

    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna("missing")

    if return_features:
        return df, df.columns.tolist()

    return df