"""Step 3 - Evaluate the detectors on the held-out patient-level test split.

For every trained detector we report
  * the native Ultralytics metrics (per-class box/mask mAP) on the test split,
  * a *label-space-independent* caries evaluation: predictions of the 9-class
    model are mapped to the unified "Caries" class so both models are scored
    on exactly the same task (caries AP50, AP50:95, lesion recall by size),
  * image-level caries screening (score = max caries confidence): AUROC,
    AUPRC and sensitivity/specificity at the threshold chosen on validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from ultralytics import YOLO

from common import CARIES_IDS9, DATA, NAMES4, NAMES9, RESULTS, read_yolo_seg, save_json


def gt_caries_boxes(label: Path, w: int, h: int):
    boxes, areas = [], []
    for c, p in read_yolo_seg(label):
        if c in CARIES_IDS9:
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            boxes.append([x0 * w, y0 * h, x1 * w, y1 * h])
            areas.append((x1 - x0) * (y1 - y0))
    return np.array(boxes).reshape(-1, 4), np.array(areas)


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


def ap_at(preds, gts, thr):
    """preds: list per image of (boxes, scores); gts: list of boxes. COCO 101-pt AP."""
    recs, n_gt = [], sum(len(g) for g in gts)
    for (pb, ps), gb in zip(preds, gts):
        order = np.argsort(-ps); pb, ps = pb[order], ps[order]
        m = iou(pb, gb); used = np.zeros(len(gb), bool)
        for i in range(len(pb)):
            tp = 0
            if len(gb):
                j = np.argmax(np.where(used, -1, m[i]))
                if not used[j] and m[i, j] >= thr:
                    used[j] = True; tp = 1
            recs.append((ps[i], tp))
    if not recs or n_gt == 0:
        return 0.0
    recs.sort(key=lambda t: -t[0])
    tp = np.cumsum([t for _, t in recs]); fp = np.cumsum([1 - t for _, t in recs])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    out = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side="left")
        out += prec[k] if k < len(prec) else 0.0
    return out / 101


def predict_caries(model, df, caries_ids, imgsz):
    preds = []
    for r in df.itertuples():
        res = model.predict(r.image, imgsz=imgsz, conf=0.001, iou=0.6, max_det=300,
                            verbose=False, retina_masks=False)[0]
        b = res.boxes
        cls = b.cls.cpu().numpy().astype(int); keep = np.isin(cls, caries_ids)
        preds.append((b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep]))
    return preds


def evaluate(weights: Path, labels: str, imgsz: int, meta: pd.DataFrame):
    model = YOLO(str(weights))
    caries_ids = CARIES_IDS9 if labels == "yolo9" else [3]
    names = NAMES9 if labels == "yolo9" else NAMES4
    tag = weights.stem
    out = {"model": tag, "labels": labels}

    # native metrics
    v = model.val(data=str(DATA / labels / "data.yaml"), split="test", imgsz=imgsz,
                  batch=4, plots=False, verbose=False)
    out["native"] = {
        "box_map50": float(v.box.map50), "box_map50_95": float(v.box.map),
        "mask_map50": float(v.seg.map50), "mask_map50_95": float(v.seg.map),
        "per_class_mask_map50": {names[c]: float(v.seg.ap50[i]) for i, c in enumerate(v.ap_class_index)},
        "per_class_box_map50": {names[c]: float(v.box.ap50[i]) for i, c in enumerate(v.ap_class_index)},
    }

    rows = []
    for split in ("val", "test"):
        d = meta[meta.split == split].reset_index(drop=True)
        preds = predict_caries(model, d, caries_ids, imgsz)
        gts, gareas = zip(*[gt_caries_boxes(Path(r.label), r.width, r.height) for r in d.itertuples()])
        if split == "test":
            out["caries_unified"] = {
                "AP50": ap_at(preds, gts, 0.5),
                "AP50_95": float(np.mean([ap_at(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)])),
            }
            # lesion recall @conf 0.25 by size tercile
            allA = np.concatenate([a for a in gareas if len(a)])
            q = np.quantile(allA, [1 / 3, 2 / 3])
            hit = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
            for (pb, ps), gb, ga in zip(preds, gts, gareas):
                pb = pb[ps >= 0.25]
                m = iou(gb, pb)
                for j in range(len(gb)):
                    k = "small" if ga[j] < q[0] else ("medium" if ga[j] < q[1] else "large")
                    hit[k][1] += 1
                    hit[k][0] += int(m.shape[1] > 0 and m[j].max() >= 0.3)
            out["caries_unified"]["recall_by_size@conf0.25,iou0.3"] = {k: v[0] / max(v[1], 1) for k, v in hit.items()}
        for r, (pb, ps) in zip(d.itertuples(), preds):
            rows.append({"stem": r.stem, "split": split, "sex": r.sex, "age_band": r.age_band,
                         "patient": r.patient, "y": r.has_caries,
                         "score": float(ps.max()) if len(ps) else 0.0})
    pr = pd.DataFrame(rows)
    va, te = pr[pr.split == "val"], pr[pr.split == "test"]
    fpr, tpr, th = roc_curve(va.y, va.score)
    thr = float(th[np.argmax(tpr - fpr)])
    yhat = (te.score >= thr).astype(int)
    out["screening"] = {
        "AUROC": roc_auc_score(te.y, te.score), "AUPRC": average_precision_score(te.y, te.score),
        "threshold_from_val": thr,
        "sensitivity": float(((yhat == 1) & (te.y == 1)).sum() / max((te.y == 1).sum(), 1)),
        "specificity": float(((yhat == 0) & (te.y == 0)).sum() / max((te.y == 0).sum(), 1)),
        "prevalence_test": float(te.y.mean()),
    }
    pr.to_csv(RESULTS / f"det_image_scores_{tag}.csv", index=False)
    save_json(out, RESULTS / f"det_metrics_{tag}.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()
    meta = pd.read_csv(DATA / "meta.csv")
    for w in a.weights:
        w = Path(w)
        if not w.exists():
            print("MISSING weights, skipped:", w); continue
        if (RESULTS / f"det_metrics_{w.stem}.json").exists():
            print("SKIP (already evaluated)", w.stem); continue
        labels = "yolo9" if w.stem.startswith("yolo9") else "yolo4"
        o = evaluate(w, labels, a.imgsz, meta)
        print(w.stem, {k: o[k] for k in ("caries_unified", "screening")})


if __name__ == "__main__":
    main()
