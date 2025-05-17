import json
import pathlib
from day5.演習3.core import load_model, load_test_data, evaluate

BASE = json.loads(pathlib.Path("metrics_baseline.json").read_text())


def test_accuracy_no_regression():
    X, y = load_test_data()
    res = evaluate(load_model(), X, y)
    assert res["accuracy"] >= BASE["accuracy"] - 1e-3


def test_inference_speed():
    X, y = load_test_data()._replace  # 例: 100 サンプルだけ計測したい場合は .head(100)
    res = evaluate(load_model(), X, y)
    assert res["infer_time"] < 0.03, f"{res['infer_time']:.3f}s is too slow"
