import yaml
from datetime import date
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Configuración de MLflow ---
mlflow.set_tracking_uri("sqlite:///mlflow.db") # dónde guardar los registros
mlflow.set_experiment("churn-telco") # agrupa las ejecuciones

# --- Parámetros (decisiones tomadas ANTES de entrenar) ---
TEST_SIZE = 0.2 # 20% para prueba, 80% para entrenar
RANDOM_STATE = 42 # semilla: hace reproducible la división
MAX_ITER = 1000 # iteraciones máximas del optimizador
C = 1.0 # inverso de la regularización
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
DVC_FILE = "data/raw/Telco-Customer-Churn.csv.dvc"

# --- 1. Leer datos y huella de DVC ---
df = pd.read_csv(DATA_PATH)
n_rows = len(df) # filas ANTES de transformar
with open(DVC_FILE) as f:
    data_md5 = yaml.safe_load(f)["outs"][0]["md5"]
    
# --- 2. Limpieza ---
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# --- 3. Texto -> columnas 0/1 ---
cols_texto = df.select_dtypes(include="str").columns
df = pd.get_dummies(df, columns=cols_texto, drop_first=True, dtype=int)

# --- 4. Separar X / y y dividir ---
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# --- 5. Nombre de la ejecución: consecutivo + fecha (ej. 09-17-10-2026) ---
runs_previos = mlflow.search_runs(
experiment_names=["churn-telco"],
run_view_type=mlflow.entities.ViewType.ALL,
)
run_name = f"{len(runs_previos) + 1:02d}-{date.today():%d-%m-%Y}"

# --- 6. Entrenar y registrar dentro de una ejecución de MLflow ---
with mlflow.start_run(run_name=run_name):
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
    max_iter=MAX_ITER, random_state=RANDOM_STATE, C=C)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_params({
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "max_iter": MAX_ITER,
    "C": C,
    })
    mlflow.log_param("n_rows", n_rows)
    mlflow.set_tag("data_md5", data_md5) # con qué datos
    mlflow.set_tag("model_type", "LogisticRegression") # qué tipo de modelo
    mlflow.log_metrics({"accuracy": acc, "f1": f1})
    mlflow.sklearn.log_model(pipeline, name="model",
    input_example=X_train.head(5))

print(f"Accuracy: {acc:.4f}")
print(f"F1 score: {f1:.4f}")