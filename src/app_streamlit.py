import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    r2_score, mean_squared_error
)

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="ML Dashboard Universel", layout="wide")
st.title("📊 Dashboard ML Universel – Classification & Régression")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models/best_model.joblib"
FEATURES_PATH = PROJECT_DIR / "models/features.json"
TASK_PATH = PROJECT_DIR / "models/task.json"



# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    [
        "🏠 Explorer Dataset",
        "⚙️ Entraîner un modèle",
        "🔮 Prédiction CSV",
        "📊 Visualisation Modèle"
    ]
)


# ========================================================
# PAGE 1 — EXPLORATION DATASET
# ========================================================
if page == "🏠 Explorer Dataset":
    uploaded = st.file_uploader("📂 Charger un CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        st.write(df.describe(include="all"))



# ========================================================
# PAGE 2 — ENTRAINEMENT DU MODELE
# ========================================================
elif page == "⚙️ Entraîner un modèle":
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression

    st.header("⚙️ Entraîner un modèle ML")

    uploaded = st.file_uploader("📂 Charger un dataset pour entraînement", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        # ---------------------------------------
        # Sélection des colonnes Y et X
        # ---------------------------------------
        st.subheader("🎯 Sélection des colonnes")
        target = st.selectbox("Colonne cible (Y)", df.columns)
        feature_cols = st.multiselect(
            "Colonnes explicatives (X)",
            df.columns.drop(target),
            default=list(df.columns.drop(target))
        )

        # ---------------------------------------
        # Colonnes numériques et catégorielles
        # ---------------------------------------
        st.subheader("🔧 Colonnes numériques & catégorielles (manuel)")

        num_features = st.multiselect(
            "Colonnes numériques",
            feature_cols,
            default=[c for c in feature_cols if df[c].dtype != "object"]
        )

        cat_features = st.multiselect(
            "Colonnes catégorielles",
            feature_cols,
            default=[c for c in feature_cols if df[c].dtype == "object"]
        )

        # ---------------------------------------
        # Choix du modèle
        # ---------------------------------------
        st.subheader("🤖 Choix du modèle ML")
        model_choice = st.selectbox(
            "Modèle",
            [
                "RandomForest (Régression)",
                "RandomForest (Classification)",
                "Régression Linéaire",
                "Logistic Regression"
            ]
        )

        if st.button("🚀 Entraîner"):
            if len(feature_cols) == 0:
                st.error("❌ Vous devez sélectionner des features.")
                st.stop()

            X = df[feature_cols]
            y = df[target]

            preprocess = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
                ]
            )

            # Modèles
            if model_choice == "RandomForest (Régression)":
                model = RandomForestRegressor()
                task = "regression"

            elif model_choice == "RandomForest (Classification)":
                model = RandomForestClassifier()
                task = "classification"

            elif model_choice == "Régression Linéaire":
                model = LinearRegression()
                task = "regression"

            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=300)
                task = "classification"

            pipeline = Pipeline([
                ("preprocessor", preprocess),
                ("model", model)
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline.fit(X_train, y_train)

            # Sauvegarde
            joblib.dump(pipeline, MODEL_PATH)
            json.dump(feature_cols, open(FEATURES_PATH, "w"))
            json.dump({"task": task}, open(TASK_PATH, "w"))

            st.success("🎉 Modèle entraîné et sauvegardé !")



# ========================================================
# PAGE 3 — PREDICTION CSV
# ========================================================
elif page == "🔮 Prédiction CSV":
    from predict import predict

    uploaded = st.file_uploader("📂 Charger un CSV", type=["csv"], key="pred")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        try:
            df_pred = predict(df)
            st.success("🎉 Prédictions générées !")
            st.dataframe(df_pred.head())

            st.download_button(
                "⬇ Télécharger les prédictions",
                df_pred.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"❌ Erreur : {e}")



# ========================================================
# PAGE 4 — VISUALISATION MODELE
# ========================================================
elif page == "📊 Visualisation Modèle":

    if not MODEL_PATH.exists():
        st.error("❌ Aucun modèle trouvé. Entraînez un modèle d'abord.")
        st.stop()

    pipeline = joblib.load(MODEL_PATH)
    FEATURES = json.load(open(FEATURES_PATH))
    TASK = json.load(open(TASK_PATH))["task"]

    st.subheader("🧠 Modèle")
    st.write(pipeline.named_steps["model"])

    st.subheader("📌 Features")
    st.write(FEATURES)

    st.subheader("🎯 Tâche")
    st.write(TASK)

    uploaded = st.file_uploader(
        "📂 Charger un CSV pour évaluation",
        type=["csv"],
        key="eval"
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        target = st.text_input("Nom de la colonne cible", "")

        if target and target in df.columns:
            X_eval = df[FEATURES]
            y_true = df[target]
            y_pred = pipeline.predict(X_eval)

            # ============================================
            # CLASSIFICATION
            # ============================================
            if TASK == "classification":
                st.subheader("📊 Matrice de confusion")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                if hasattr(pipeline.named_steps["model"], "predict_proba"):
                    st.subheader("📈 ROC Curve")
                    y_proba = pipeline.predict_proba(X_eval)[:, 1]
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    auc_score = auc(fpr, tpr)
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
                    ax2.plot([0,1], [0,1], "k--")
                    st.pyplot(fig2)

            # ============================================
            # REGRESSION
            # ============================================
            else:
                st.subheader("📈 Réel vs Prédit")
                fig3, ax3 = plt.subplots()
                ax3.scatter(y_true, y_pred)
                ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
                ax3.set_title(f"R2 = {r2_score(y_true, y_pred):.2f}")
                st.pyplot(fig3)

                st.subheader("📊 Résidus")
                residuals = y_true - y_pred
                fig4, ax4 = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax4)
                ax4.set_title(f"RMSE = {mean_squared_error(y_true, y_pred):.2f}")
                st.pyplot(fig4)
