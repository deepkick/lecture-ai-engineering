# core.py  --- 再利用しやすいユーティリティ
import time, joblib, pandas as pd
from sklearn.metrics import accuracy_score

MODEL_PATH = "./models/titanic_model.pkl"


def load_test_data():
    df = pd.read_csv("./data/Titanic.csv")  # 必要に応じてコピー
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
