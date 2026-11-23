"""Saliency normalisation, plausibility scores and Grad-CAM mechanics on a tiny CPU model."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
import xai_eval  # noqa: E402
from xai_eval import CAM, S, norm, plaus  # noqa: E402


def test_norm():
    s = norm(np.array([-1.0, 0.0, 2.0, 4.0]))
    assert np.allclose(s, [0, 0, 0.5, 1.0])
    assert np.all(norm(np.zeros(5)) == 0)


def test_plausibility_scores_reward_saliency_on_the_lesion():
    mask = np.zeros((S, S), np.uint8)
    mask[100:130, 100:130] = 1
    on = np.zeros((S, S)); on[100:130, 100:130] = 1.0
    off = np.zeros((S, S)); off[10:20, 10:20] = 1.0
    good, bad = plaus(on, mask), plaus(off, mask)
    assert good["pointing"] == 1.0 and bad["pointing"] == 0.0
    assert good["mass"] == pytest.approx(1.0) and bad["mass"] == pytest.approx(0.0)
    assert good["lift"] == pytest.approx(1 / mask.mean())
    assert good["pix_auroc"] > 0.99 and bad["pix_auroc"] < 0.6


def test_gradcam_on_tiny_model(monkeypatch):
    monkeypatch.setattr(xai_eval, "DEV", "cpu")
    torch.manual_seed(0)
    feat = torch.nn.Conv2d(3, 4, 3, stride=8, padding=1)
    model = torch.nn.Sequential(feat, torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(4, 1)).eval()
    x = torch.randn(1, 3, S, S)
    for plus in (False, True):
        cam = CAM(model, model[1])(x, plus=plus)
        assert cam.shape == (S, S)
        assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
