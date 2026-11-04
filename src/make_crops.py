"""Step 4 - Build a tooth-region crop dataset for screening + explainability.

Every crop keeps its pixel-accurate caries mask, which is what makes a
*quantitative* explainability audit possible later (saliency vs. lesion).

Crop types (per image, in a frame resized to LONG=1280 px, crop CROP=256 px):
  pos       - centred on a caries lesion with random jitter (lesion stays inside)
  hard_neg  - centred on a filling / crown / abrasion with *zero* caries pixels
  rand_neg  - random location with zero caries pixels
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import CARIES_IDS9, DATA, read_yolo_seg

LONG, CROP, OUT = 1280, 256, 224


def rasterise(polys, w, h, ids):
    m = np.zeros((h, w), np.uint8)
    for c, p in polys:
        if c in ids:
            cv2.fillPoly(m, [np.round(p * [w, h]).astype(np.int32)], 1)
    return m


def clip_box(cx, cy, w, h):
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP)); y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
    return x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pos", type=int, default=8)
    ap.add_argument("--n_hard", type=int, default=3)
    ap.add_argument("--n_rand", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta = pd.read_csv(DATA / "meta.csv")
    out = DATA / "crops"; (out / "img").mkdir(parents=True, exist_ok=True); (out / "mask").mkdir(exist_ok=True)
    rows = []
    for r in tqdm(meta.itertuples(), total=len(meta)):
        img = cv2.imread(r.image)
        s = LONG / max(img.shape[:2])
        if abs(s - 1) > 1e-3:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        polys = read_yolo_seg(__import__("pathlib").Path(r.label))
        cm = rasterise(polys, w, h, CARIES_IDS9)
        cands = []
        car = [p for c, p in polys if c in CARIES_IDS9]
        rng.shuffle(car)
        for p in car[: a.max_pos]:
            px = p * [w, h]; (x0, y0), (x1, y1) = px.min(0), px.max(0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            jx = rng.uniform(-1, 1) * max(0, (CROP - (x1 - x0)) / 2 - 8)
            jy = rng.uniform(-1, 1) * max(0, (CROP - (y1 - y0)) / 2 - 8)
            cands.append(("pos", cx + jx, cy + jy))
        other = [p for c, p in polys if c not in CARIES_IDS9]
        rng.shuffle(other)
        nh = 0
        for p in other:
            if nh >= a.n_hard:
                break
            c = (p * [w, h]).mean(0)
            x0, y0 = clip_box(*c, w, h)
            if cm[y0:y0 + CROP, x0:x0 + CROP].sum() == 0:
                cands.append(("hard_neg", *c)); nh += 1
        nr, tries = 0, 0
        while nr < a.n_rand and tries < 50:
            tries += 1
            cx, cy = rng.uniform(CROP / 2, w - CROP / 2), rng.uniform(CROP / 2, h - CROP / 2)
            x0, y0 = clip_box(cx, cy, w, h)
            if cm[y0:y0 + CROP, x0:x0 + CROP].sum() == 0:
                cands.append(("rand_neg", cx, cy)); nr += 1
        for k, (kind, cx, cy) in enumerate(cands):
            x0, y0 = clip_box(cx, cy, w, h)
            ci = cv2.resize(img[y0:y0 + CROP, x0:x0 + CROP], (OUT, OUT), interpolation=cv2.INTER_AREA)
            mi = cv2.resize(cm[y0:y0 + CROP, x0:x0 + CROP], (OUT, OUT), interpolation=cv2.INTER_NEAREST)
            name = f"{r.stem}_{k:02d}"
            cv2.imwrite(str(out / "img" / f"{name}.jpg"), ci, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / "mask" / f"{name}.png"), mi * 255)
            rows.append({"crop": name, "stem": r.stem, "kind": kind, "y": int(mi.sum() > 0),
                         "mask_frac": float(mi.mean()), "split": r.split, "patient": r.patient,
                         "sex": r.sex, "age": r.age, "age_band": r.age_band})
    df = pd.DataFrame(rows)
    df.to_csv(out / "crops.csv", index=False)
    print(df.groupby(["split", "kind"]).size().unstack())


if __name__ == "__main__":
    main()
