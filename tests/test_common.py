import numpy as np
import pytest

from common import CARIES_IDS9, MAP9TO4, NAMES4, NAMES9, age_band, load_json, parse_name, read_yolo_seg, save_json


def test_parse_name():
    assert parse_name("p012_F_37_004") == {"patient": "p012", "sex": "F", "age": 37, "shot": 4}


def test_parse_name_rejects_unexpected_stems():
    with pytest.raises(ValueError):
        parse_name("IMG_0001")


@pytest.mark.parametrize("age,band", [(18, "<30"), (29, "<30"), (30, "30-44"), (44, "30-44"), (45, ">=45"), (80, ">=45")])
def test_age_band_boundaries(age, band):
    assert age_band(age) == band


def test_label_spaces_are_consistent():
    assert len(NAMES9) == 9 and len(NAMES4) == 4
    assert set(MAP9TO4) == set(range(9))
    # every Black class collapses onto "Caries", the non-caries classes keep their id
    assert {MAP9TO4[c] for c in CARIES_IDS9} == {NAMES4.index("Caries")}
    assert all(MAP9TO4[c] == c for c in range(9) if c not in CARIES_IDS9)


def test_read_yolo_seg(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("3 0.1 0.1 0.2 0.1 0.2 0.2\n"
                 "1 0.5 0.5\n"  # too short for a polygon -> skipped
                 "0 0 0 1 0 1 1 0 1\n")
    out = read_yolo_seg(f)
    assert [c for c, _ in out] == [3, 0]
    assert out[0][1].shape == (3, 2) and out[1][1].shape == (4, 2)
    assert out[0][1].dtype == np.float32


def test_read_yolo_seg_missing_file_is_empty(tmp_path):
    assert read_yolo_seg(tmp_path / "missing.txt") == []


def test_json_roundtrip(tmp_path):
    obj = {"auc": np.float32(0.75), "n": 3, "names": ["a", "b"]}
    p = tmp_path / "sub" / "m.json"
    save_json(obj, p)
    back = load_json(p)
    assert back["n"] == 3 and back["names"] == ["a", "b"]
    assert back["auc"] == pytest.approx(0.75)
