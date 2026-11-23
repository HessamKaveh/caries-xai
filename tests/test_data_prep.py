"""Patient-level splitting, polygon geometry and crop helpers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from make_crops import CROP, clip_box, rasterise
from prepare_data import polygon_area, split_patients

ROOT = Path(__file__).resolve().parents[1]


def test_polygon_area():
    assert polygon_area(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)) == pytest.approx(1.0)
    assert polygon_area(np.array([[0, 0], [0, 1], [1, 1], [1, 0]], float)) == pytest.approx(1.0)  # orientation-free
    assert polygon_area(np.array([[0, 0], [4, 0], [0, 3]], float)) == pytest.approx(6.0)


def test_split_patients_is_patient_disjoint():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(120):
        sex, band = rng.choice(["M", "F"]), rng.choice(["<30", "30-44", ">=45"])
        rows += [{"patient": f"p{k:03d}", "sex": sex, "age_band": band}] * int(rng.integers(1, 6))
    df = split_patients(pd.DataFrame(rows), seed=42)
    assert df.groupby("patient").split.nunique().max() == 1
    assert set(df.split) == {"train", "val", "test"}
    frac = df.drop_duplicates("patient").split.value_counts(normalize=True)
    assert frac["test"] == pytest.approx(0.2, abs=0.05)
    assert frac["val"] == pytest.approx(0.1, abs=0.05)


def test_clip_box_stays_inside_image():
    assert clip_box(500, 400, 1280, 853) == (500 - CROP // 2, 400 - CROP // 2)
    assert clip_box(3, 3, 1280, 853) == (0, 0)
    assert clip_box(1279, 852, 1280, 853) == (1280 - CROP, 853 - CROP)


def test_rasterise_selects_classes():
    sq = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], np.float32)
    polys = [(4, sq), (1, sq + 0.5)]
    m = rasterise(polys, 100, 100, ids=[4])
    assert m.dtype == np.uint8 and m.shape == (100, 100)
    assert m[10, 10] == 1 and m[80, 80] == 0
    assert 0.24 < m.mean() < 0.28


@pytest.mark.skipif(not (ROOT / "data" / "meta.csv").exists(),
                    reason="needs the prepared AlphaDent split (data/meta.csv, not distributed; see README)")
def test_prepared_split_is_patient_disjoint():
    meta = pd.read_csv(ROOT / "data" / "meta.csv")
    assert meta.groupby("patient").split.nunique().max() == 1


def test_published_logits_are_patient_disjoint():
    """The committed per-crop logits must respect the patient-level split they claim."""
    files = sorted((ROOT / "results").glob("cls_logits_*_patient_*.csv"))
    if not files:
        pytest.skip("no committed classifier logits in results/")
    by_split = {}
    for f in files:
        d = pd.read_csv(f, usecols=["patient", "split"])
        for s, g in d.groupby("split"):
            by_split.setdefault(s, set()).update(g.patient)
    assert not (by_split.get("val", set()) & by_split.get("test", set()))
