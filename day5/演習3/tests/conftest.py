import sys
from pathlib import Path

# ルート (lecture-ai-engineering/) を確実に import path へ追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
