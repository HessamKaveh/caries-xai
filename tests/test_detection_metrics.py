"""Box IoU and COCO-style AP used for the caries detector evaluation (src/eval_detection.py)."""
import numpy as np
import pytest

pytest.importorskip("ultralytics")  # eval_detection imports YOLO at module level
from eval_detection import ap_at, gt_caries_boxes, iou  # noqa: E402


def test_iou_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-6)


def test_iou_empty():
    assert iou(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_ap_perfect_and_empty():
    gts = [np.array([[0, 0, 10, 10]], float), np.array([[5, 5, 20, 20]], float)]
    preds = [(g.copy(), np.array([0.9])) for g in gts]
    assert ap_at(preds, gts, 0.5) == pytest.approx(1.0)
    assert ap_at([(np.zeros((0, 4)), np.zeros(0))] * 2, gts, 0.5) == 0.0


def test_ap_penalises_high_scoring_false_positive():
    gts = [np.array([[0, 0, 10, 10]], float)]
    pb = np.array([[50, 50, 60, 60], [0, 0, 10, 10]], float)
    good = ap_at([(pb, np.array([0.1, 0.9]))], gts, 0.5)
    bad = ap_at([(pb, np.array([0.9, 0.1]))], gts, 0.5)
    assert good == pytest.approx(1.0)
    assert bad == pytest.approx(0.5, abs=0.01)


def test_ap_one_prediction_cannot_match_two_gts():
    gts = [np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)]
    preds = [(np.array([[0, 0, 10, 10]], float), np.array([0.9]))]
    assert ap_at(preds, gts, 0.5) == pytest.approx(51 / 101)  # recall capped at 0.5


def test_gt_caries_boxes_keeps_only_caries(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("1 0.1 0.1 0.2 0.1 0.2 0.2\n"           # filling -> ignored
                 "4 0.5 0.5 0.75 0.5 0.75 0.75 0.5 0.75\n")  # caries class 2
    boxes, areas = gt_caries_boxes(f, 200, 100)
    assert np.allclose(boxes, [[100, 50, 150, 75]])
    assert np.allclose(areas, [0.0625])
