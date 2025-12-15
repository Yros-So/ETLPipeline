import pandas as pd

class BasicFeatureEngineer:

    @staticmethod
    def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # Nettoyage des colonnes object
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].astype(str).str.strip()

        # Convertir colonnes bool en string
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(str)

        # Remplacement valeurs incohérentes
        df = df.replace({
            'nan': None, 'NaN': None, 'None': None, 'none': None
        })

        # Création feature TotalServices
        services_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"
        ]

        def count_services(row):
            cnt = 0
            for c in services_cols:
                v = row.get(c, "No")
                if isinstance(v, str) and v.lower().startswith("yes"):
                    cnt += 1
            return cnt

        df["TotalServices"] = df.apply(count_services, axis=1)

        # IMPORTANT : NE PAS convertir toutes les colonnes en string
        # sklearn a besoin des colonnes numériques
        return df 