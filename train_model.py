import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

# MODELES CLASSIFICATION
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# MODELES REGRESSION
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Optional : XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False


# ============================================================
# 🚀 FONCTION POUR OBTENIR LE BON MODELE SELON LA TÂCHE
# ============================================================
def get_model(task="classification"):

    if task == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=150),
            "svm": SVC(probability=True),
        }
        if HAS_XGB:
            models["xgboost"] = XGBClassifier(eval_metric="logloss")
        return models

    elif task == "regression":
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=150),
            "svr": SVR(),
        }
        if HAS_XGB:
            models["xgboost_regressor"] = XGBRegressor()
        return models

    else:
        raise ValueError("Task inconnue. Choisir classification ou regression.")


# ============================================================
# 📌 PARAMÈTRES GÉNÉRAUX 
# ============================================================
DATA_FILE = "data/train.csv"
TARGET = "Tenure"   # À modifier selon dataset

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURES_PATH = MODELS_DIR / "features.json"
TASK_PATH = MODELS_DIR / "task.json"


# ============================================================
# 📥 CHARGEMENT DU DATASET
# ============================================================
df = pd.read_csv(DATA_FILE)

y = df[TARGET]
X = df.drop(columns=[TARGET])


# ============================================================
# 🔍 DETECTION AUTOMATIQUE DU TYPE DE TÂCHE
# ============================================================
if y.dtype == "object" or y.nunique() <= 20:
    task = "classification"
else:
    task = "regression"

print(f"🧠 Tâche détectée : {task.upper()}")


# ============================================================
# 🎨 ENCODAGE & SCALING (Universel)
# ============================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


# ============================================================
# 🧪 TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = get_model(task)
best_pipeline = None
best_score = -999


print("\n🚀 Entraînement des modèles disponibles...\n")

# ============================================================
# 🔁 TESTER TOUS LES MODELES POSSIBLES
# ============================================================
for model_name, model_obj in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_obj)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if task == "classification":
        score = f1_score(y_test, y_pred, average="weighted")
    else:
        score = r2_score(y_test, y_pred)

    print(f"📌 {model_name} → Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_pipeline = pipeline
        best_model_name = model_name


print("\n🏆 Best Model :", best_model_name)
print("📈 Best Score :", round(best_score, 4))


# ============================================================
# 💾 SAUVEGARDE DU MEILLEUR PIPELINE
# ============================================================
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\n✅ Modèle sauvegardé → {MODEL_PATH}")

with open(FEATURES_PATH, "w") as f:
    json.dump(list(X.columns), f)

with open(TASK_PATH, "w") as f:
    json.dump({"task": task}, f)

print(f"📁 Features sauvegardées dans → {FEATURES_PATH}")
print(f"📁 Type de tâche sauvegardé dans → {TASK_PATH}")
