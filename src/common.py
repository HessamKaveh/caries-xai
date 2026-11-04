"""Shared constants and helpers for the CariesXAI pipeline."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("CARIES_RAW", Path.home() / "caries-xai-data/raw/alphadent"))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"

# Original AlphaDent label space (Black's classification for caries).
NAMES9 = [
    "Abrasion", "Filling", "Crown",
    "Caries 1 class", "Caries 2 class", "Caries 3 class",
    "Caries 4 class", "Caries 5 class", "Caries 6 class",
]
CARIES_IDS9 = [3, 4, 5, 6, 7, 8]
# Clinically unified label space proposed in this project.
NAMES4 = ["Abrasion", "Filling", "Crown", "Caries"]
MAP9TO4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}

FNAME_RE = re.compile(r"(p\d+)_([MF])_(\d+)_(\d+)")


def parse_name(stem: str) -> dict:
    m = FNAME_RE.match(stem)
    if not m:
        raise ValueError(f"unexpected file name {stem}")
    pid, sex, age, idx = m.groups()
    return {"patient": pid, "sex": sex, "age": int(age), "shot": int(idx)}


def age_band(age: int) -> str:
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    return ">=45"


def read_yolo_seg(path: Path) -> list[tuple[int, np.ndarray]]:
    """Return list of (class_id, polygon[N,2] in normalised xy)."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        cls = int(v[0])
        pts = np.asarray(v[1:], dtype=np.float32).reshape(-1, 2)
        out.append((cls, pts))
    return out


def seed_all(seed: int = 0):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path):
    return json.loads(Path(path).read_text())
