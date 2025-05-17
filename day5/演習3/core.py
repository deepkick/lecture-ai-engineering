# core.py  --- 再利用しやすいユーティリティ
import time
import joblib
import pandas as pd
import pathlib
from sklearn.metrics import accuracy_score

MODEL_PATH = "./models/titanic_model.pkl"


ROOT = pathlib.Path(__file__).parent  # day5/演習3/


def load_test_data():
    csv_path = ROOT / "data" / "Titanic.csv"
    df = pd.read_csv(csv_path)
    X, y = df.drop("Survived", axis=1), df["Survived"]
    return X, y


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def evaluate(model, X, y):
    t0 = time.perf_counter()
    preds = model.predict(X)
    per_sample = (time.perf_counter() - t0) / len(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "infer_time": per_sample,
    }
