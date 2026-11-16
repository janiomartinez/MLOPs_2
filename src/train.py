import joblib
import pandas as pd
import yaml
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# --- Configuración ---
TRAIN_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.joblib")

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]
    
# --- Datos ---
train_df = pd.read_csv(TRAIN_PATH)
X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
# Separar columnas de texto y numéricas
cols_texto = X_train.select_dtypes(include="str").columns.tolist()
cols_numericas = X_train.select_dtypes(include="number").columns.tolist()
# --- Preprocesamiento ---
preprocesador = ColumnTransformer([
    ("texto", OneHotEncoder(drop="first", handle_unknown="ignore",sparse_output=False), cols_texto),("numericas", "passthrough", cols_numericas),
])

# --- Entrenamiento ---
pipeline = Pipeline([
    ("preprocesador", preprocesador), # texto -> columnas 0/1
    ("scaler", StandardScaler()), # escala todas las columnas
    ("classifier", LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        random_state=params["random_state"],
    )),
])

pipeline.fit(X_train, y_train)

# --- Guardar modelo ---
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Modelo guardado en {MODEL_PATH}")
    
