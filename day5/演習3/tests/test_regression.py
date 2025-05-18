"""
regression tests:
1. accuracy does not degrade compared with metrics_baseline.json
2. inference speed is not slower than 1.2 × baseline
"""

import json
import pathlib
import time

from day5.演習3.core import load_model, load_test_data

# ── ベースライン指標をロード ──────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root
BASE = json.loads((ROOT / "metrics_baseline.json").read_text())


# ── 1. 精度リグレッションテスト ─────────────────────
def test_accuracy_no_regression() -> None:
    X, y = load_test_data()
    model = load_model()
    acc = (model.predict(X) == y).mean()
    assert acc >= BASE["accuracy"]


# ── 2. 推論速度リグレッションテスト ──────────────────
def test_inference_speed() -> None:
    X, _ = load_test_data()
    X = X.head(100)  # 軽量計測
    start = time.perf_counter()
    _ = load_model().predict(X)
    per_sample = (time.perf_counter() - start) / len(X)

    # ベースラインより 20 % 以上遅くなっていないか
    assert per_sample <= BASE["infer_time"] * 1.2
