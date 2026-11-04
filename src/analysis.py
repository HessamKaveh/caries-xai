"""Step 7 - Calibration, selective prediction ("refer to dentist"), subgroup
fairness, leakage quantification, bootstrap CIs, figures and RESULTS.md."""
from __future__ import annotations

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from common import FIG, RESULTS, load_json, save_json

ARCHS = ["resnet50", "efficientnet_b0"]
rng = np.random.default_rng(0)


def sig(z):
    return 1 / (1 + np.exp(-z))


def nll(T, z, y):
    p = np.clip(sig(z / T), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(p, y, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def patient_bootstrap(df, score, y="y", B=1000):
    """95 % CI for AUROC resampling *patients* (clustered data)."""
    pats = df.patient.unique(); groups = {p: g for p, g in df.groupby("patient")}
    vals = []
    for _ in range(B):
        d = pd.concat([groups[p] for p in rng.choice(pats, len(pats))])
        if d[y].nunique() == 2:
            vals.append(roc_auc_score(d[y], d[score]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def risk_coverage(p, y):
    conf = np.maximum(p, 1 - p); pred = (p >= 0.5).astype(int)
    order = np.argsort(-conf); err = (pred[order] != y[order]).astype(float)
    cov = np.arange(1, len(y) + 1) / len(y); risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    return cov, risk, float(np.mean(risk))


def classifier_block():
    res, curves = {}, {}
    for arch in ARCHS:
        va = pd.read_csv(RESULTS / f"cls_logits_{arch}_patient_val.csv")
        te = pd.read_csv(RESULTS / f"cls_logits_{arch}_patient_test.csv")
        T = minimize_scalar(nll, bounds=(0.05, 20), args=(va.logit.values, va.y.values), method="bounded").x
        p_raw, p_cal = sig(te.logit.values), sig(te.logit.values / T); y = te.y.values
        fpr, tpr, th = roc_curve(va.y, va.logit); thr = th[np.argmax(tpr - fpr)]
        yhat = (te.logit.values >= thr).astype(int)
        cov, risk, aurc = risk_coverage(p_cal, y)
        r = {
            "AUROC": roc_auc_score(y, te.logit), "AUROC_95CI_patient_bootstrap": patient_bootstrap(te, "logit"),
            "AUPRC": average_precision_score(y, te.logit),
            "sensitivity": float(yhat[y == 1].mean()), "specificity": float(1 - yhat[y == 0].mean()),
            "temperature": float(T), "ECE_raw": ece(p_raw, y), "ECE_calibrated": ece(p_cal, y),
            "AURC": aurc,
        }
        for c in (1.0, 0.9, 0.8, 0.7):
            k = int(c * len(y)) - 1
            r[f"error@coverage{int(c*100)}"] = float(risk[k])
        # AUROC by crop type: can it separate caries from restorations?
        hard = te[te.kind.isin(["pos", "hard_neg"])]
        r["AUROC_caries_vs_restorations"] = roc_auc_score(hard.y, hard.logit)
        # subgroups
        sub = {}
        for col in ("sex", "age_band"):
            for g, d in te.groupby(col):
                if d.y.nunique() == 2:
                    sub[f"{col}={g}"] = {"n": int(len(d)), "AUROC": roc_auc_score(d.y, d.logit),
                                         "CI95": patient_bootstrap(d, "logit", B=500)}
        r["subgroups"] = sub
        # leakage experiment
        lk = RESULTS / f"cls_train_{arch}_random.json"
        if lk.exists():
            r["leaky_random_split_test_AUROC"] = load_json(lk)["test_auroc"]
        res[arch] = r
        curves[arch] = dict(te=te, p_raw=p_raw, p_cal=p_cal, cov=cov, risk=risk)
    return res, curves


def detector_block():
    res = {}
    for f in sorted(glob.glob(str(RESULTS / "det_metrics_*.json"))):
        m = load_json(f); tag = m["model"]
        sc = pd.read_csv(RESULTS / f"det_image_scores_{tag}.csv"); te = sc[sc.split == "test"]
        m["screening"]["AUROC_95CI_patient_bootstrap"] = patient_bootstrap(te, "score")
        m["subgroups"] = {f"{col}={g}": {"n": int(len(d)), "AUROC": roc_auc_score(d.y, d.score)}
                          for col in ("sex", "age_band") for g, d in te.groupby(col) if d.y.nunique() == 2}
        res[tag] = m
    return res


def figures(curves, det):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for arch, c in curves.items():
        fpr, tpr, _ = roc_curve(c["te"].y, c["te"].logit)
        ax[0].plot(fpr, tpr, label=f"{arch} (crop) AUROC={roc_auc_score(c['te'].y, c['te'].logit):.3f}")
    for tag in det:
        sc = pd.read_csv(RESULTS / f"det_image_scores_{tag}.csv"); te = sc[sc.split == "test"]
        fpr, tpr, _ = roc_curve(te.y, te.score)
        ax[0].plot(fpr, tpr, "--", label=f"{tag.split('_')[0]} detector (image) AUROC={roc_auc_score(te.y, te.score):.3f}")
    ax[0].plot([0, 1], [0, 1], ":", c="grey"); ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity")
    ax[0].set_title("Caries screening ROC (patient-level test)"); ax[0].legend(fontsize=7)
    for arch, c in curves.items():
        for key, ls in (("p_raw", ":"), ("p_cal", "-")):
            p, y = c[key], c["te"].y.values; e = np.linspace(0, 1, 11); xs, ys = [], []
            for lo, hi in zip(e[:-1], e[1:]):
                m = (p > lo) & (p <= hi)
                if m.sum() > 5:
                    xs.append(p[m].mean()); ys.append(y[m].mean())
            ax[1].plot(xs, ys, ls, marker="o", ms=3, label=f"{arch} {'calibrated' if key == 'p_cal' else 'raw'}")
    ax[1].plot([0, 1], [0, 1], c="grey", lw=0.8); ax[1].set_title("Reliability diagram")
    ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed frequency"); ax[1].legend(fontsize=7)
    for arch, c in curves.items():
        ax[2].plot(c["cov"], c["risk"], label=arch)
    ax[2].set_xlabel("coverage (fraction decided automatically)"); ax[2].set_ylabel("error rate")
    ax[2].set_title("Risk-coverage: defer uncertain crops to dentist"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "screening_calibration_selective.png", dpi=150); plt.close(fig)

    xs = RESULTS / "xai_summary.csv"
    if xs.exists():
        s = pd.read_csv(xs)
        mets = [("pointing", "Pointing game ↑"), ("pix_auroc", "Pixel AUROC vs mask ↑"),
                ("deletion_auc", "Deletion AUC ↓"), ("insertion_auc", "Insertion AUC ↑")]
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
        order = ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Occlusion", "Centre prior", "Random"]
        w = 0.38
        for k, (m, t) in enumerate(mets):
            for j, arch in enumerate(ARCHS):
                d = s[s.arch == arch].set_index("method").reindex(order)
                ax[k].bar(np.arange(len(order)) + (j - 0.5) * w, d[m].values, w, label=arch)
            ax[k].set_xticks(range(len(order))); ax[k].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
            ax[k].set_title(t)
        ax[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "xai_audit.png", dpi=150); plt.close(fig)


def main():
    cls, curves = classifier_block()
    det = detector_block()
    save_json({"classifiers": cls, "detectors": det}, RESULTS / "final_metrics.json")
    figures(curves, det)
    print(pd.DataFrame({a: {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for a, r in cls.items()}).round(4))
    for t, m in det.items():
        print(t, m["caries_unified"], m["screening"])


if __name__ == "__main__":
    main()
