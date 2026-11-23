"""Calibration, selective prediction and clustered-bootstrap helpers (src/analysis.py)."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from analysis import ece, nll, patient_bootstrap, risk_coverage, sig


def test_sigmoid():
    assert sig(0.0) == 0.5
    assert np.allclose(sig(np.array([-50.0, 50.0])), [0.0, 1.0])
    z = np.linspace(-5, 5, 11)
    assert np.allclose(sig(z) + sig(-z), 1.0)


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.25, 0.75], 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y, bins=10) == pytest.approx(0.0)


def test_ece_of_overconfident_predictor():
    p = np.full(10, 0.9)
    y = np.array([1] * 5 + [0] * 5)
    assert ece(p, y) == pytest.approx(0.4)


def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 2, 20000)
    y = (rng.random(z_true.size) < sig(z_true)).astype(float)
    z_over = 3.0 * z_true  # an over-confident model: logits inflated by T = 3
    T = minimize_scalar(nll, bounds=(0.05, 20), args=(z_over, y), method="bounded").x
    assert T == pytest.approx(3.0, rel=0.05)
    assert ece(sig(z_over / T), y) < ece(sig(z_over), y)


def test_risk_coverage():
    # confident predictions are right, the least confident one is wrong
    p = np.array([0.99, 0.02, 0.9, 0.45])
    y = np.array([1, 0, 1, 1])
    cov, risk, aurc = risk_coverage(p, y)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0, 0, 0, 0.25])
    assert aurc == pytest.approx(0.0625)


def test_patient_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    rows = []
    for k in range(40):
        y = rng.integers(0, 2, 5)
        rows.append(pd.DataFrame({"patient": f"p{k}", "y": y, "s": y + rng.normal(0, 1, 5)}))
    df = pd.concat(rows, ignore_index=True)
    lo, hi = patient_bootstrap(df, "s", B=200)
    auc = roc_auc_score(df.y, df.s)
    assert 0.5 < lo < auc < hi <= 1.0
