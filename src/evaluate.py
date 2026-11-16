import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from datetime import date
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# --- Configuración ---
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("metrics.json")
DVC_FILE = Path("data/raw/Telco-Customer-Churn.csv.dvc")

with open("params.yaml") as f:
    params = yaml.safe_load(f)
with open(DVC_FILE) as f:
    data_md5 = yaml.safe_load(f)["outs"][0]["md5"]
    
# --- Evaluación ---
test_df = pd.read_csv(TEST_PATH)
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]
model = joblib.load(MODEL_PATH)
y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}

METRICS_PATH.write_text(json.dumps(metrics, indent=2))
# Filas totales del dataset (train + test)
n_rows = len(pd.read_csv(TRAIN_PATH)) + len(test_df)
# --- Registro en MLflow ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn-telco")
# Nombre de la ejecución: consecutivo + fecha (ej. 13-17-10-2026)

runs_previos = mlflow.search_runs(
    experiment_names=["churn-telco"],
    run_view_type=mlflow.entities.ViewType.ALL,
)
run_name = f"{len(runs_previos) + 1:02d}-{date.today():%d-%m-%Y}"

with mlflow.start_run(run_name=run_name):
    # Qué decidimos
    mlflow.log_params({
        "test_size": params["prepare"]["test_size"],
        "random_state": params["train"]["random_state"],
        "max_iter": params["train"]["max_iter"],
        "C": params["train"]["C"],
        "n_rows": n_rows,
    })
    # Con qué datos y qué modelo
    mlflow.set_tag("data_md5", data_md5)
    mlflow.set_tag("model_type", "LogisticRegression")
    # Qué salió
    mlflow.log_metrics(metrics)
    # El modelo entrenado
    mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(5))
    
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 score: {metrics['f1']:.4f}")