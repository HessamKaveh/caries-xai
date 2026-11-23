import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The pipeline scripts in src/ import each other as top-level modules (`from common import ...`).
sys.path.insert(0, str(ROOT / "src"))
