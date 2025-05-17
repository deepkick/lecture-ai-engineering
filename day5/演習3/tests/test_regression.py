# tests/test_regression.py (簡易版)
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


def test_dummy_accuracy():
    X = pd.DataFrame({"a": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    clf = DummyClassifier(strategy="most_frequent").fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    assert acc >= 0.25  # しきい値はお好みで
