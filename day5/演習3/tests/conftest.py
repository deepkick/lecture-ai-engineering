"""
テスト開始時にリポジトリ直下(lecture-ai-engineering/)を PYTHONPATH に追加する。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # tests → 演習3 → day5 → ★ルート
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
