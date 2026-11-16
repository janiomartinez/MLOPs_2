import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
# --- Configuración ---
INPUT_PATH = Path("data/raw/Telco-Customer-Churn.csv")
OUTPUT_DIR = Path("data/processed")
with open("params.yaml") as f:
    params = yaml.safe_load(f)["prepare"] # solo la sección prepare
    
# --- Limpieza ---
# 1. Leer el dataset original
df = pd.read_csv(INPUT_PATH)    
# 2. Quitar customerID (identificador, no aporta para predecir)
df = df.drop(columns=["customerID"])
# 3. Convertir la variable objetivo Churn a 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
# 4. Corregir TotalCharges: texto -> número; los vacíos pasan a 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# --- División train/test ---
# stratify mantiene la misma proporción de Churn en ambos conjuntos
train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["Churn"],
)

# --- Guardar resultado ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
print(f"train.csv: {train_df.shape} | test.csv: {test_df.shape}")