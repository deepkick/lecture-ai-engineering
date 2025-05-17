# main.py  --- 手元実行／ベースライン生成用ワンショットスクリプト
import json
import pathlib
from core import load_model, load_test_data, evaluate

if __name__ == "__main__":
    X, y = load_test_data()
    metrics = evaluate(load_model(), X, y)
    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['infer_time']:.4f} 秒/サンプル")

    # 初回にベースラインを保存する
    pathlib.Path("metrics_baseline.json").write_text(json.dumps(metrics, indent=2))
