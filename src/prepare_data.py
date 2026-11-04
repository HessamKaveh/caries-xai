"""Step 1 - Build the patient-level, demographically stratified benchmark.

* Merges the officially labelled AlphaDent images (train + valid; the official
  test split has no public labels) and re-splits them **by patient** into
  train / val / test (~70 / 10 / 20 %), stratified on sex x age band so that
  every split has the same demographic mix.
* Writes two YOLO-seg datasets that share images and splits:
    - data/yolo9 : original 9 classes (Black's caries classes 1-6 kept apart)
    - data/yolo4 : clinically unified 4 classes (all caries -> "Caries")
* Writes data/meta.csv (one row per image) and dataset statistics/figures.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from common import (CARIES_IDS9, DATA, FIG, MAP9TO4, NAMES4, NAMES9, RAW, RESULTS,
                    age_band, parse_name, read_yolo_seg, save_json)


LONG = 1280


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def collect() -> pd.DataFrame:
    rows = []
    for split in ("train", "valid"):
        for img in sorted((RAW / "images" / split).iterdir()):
            stem = img.stem
            lab = RAW / "labels" / split / f"{stem}.txt"
            polys = read_yolo_seg(lab)
            # cache a 1280-px copy: 18 MP originals make JPEG decoding the bottleneck
            small = DATA / "img1280" / f"{stem}.jpg"
            if not small.exists():
                small.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(img) as im:
                    im = im.convert("RGB"); im.thumbnail((LONG, LONG), Image.LANCZOS)
                    im.save(small, quality=95)
            with Image.open(small) as im:
                w, h = im.size
            cnt = Counter(c for c, _ in polys)
            caries_area = sum(polygon_area(p) for c, p in polys if c in CARIES_IDS9)
            r = {"image": str(small), "orig_image": str(img), "label": str(lab), "stem": stem, "orig_split": split,
                 "width": w, "height": h, "n_inst": len(polys),
                 "n_caries": sum(cnt[c] for c in CARIES_IDS9),
                 "caries_area_frac": caries_area}
            r.update(parse_name(stem))
            for i, n in enumerate(NAMES9):
                r[f"n_{i}"] = cnt.get(i, 0)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["has_caries"] = (df.n_caries > 0).astype(int)
    df["age_band"] = df.age.map(age_band)
    return df


def split_patients(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    strata = (df.sex + "_" + df.age_band).values
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(df, strata, groups=df.patient)):
        fold[te] = k
    df["fold"] = fold
    df["split"] = np.where(fold <= 1, "test", np.where(fold == 2, "val", "train"))
    # sanity: no patient in two splits
    assert df.groupby("patient").split.nunique().max() == 1
    return df


def write_yolo(df: pd.DataFrame, name: str, mapping: dict | None, names: list[str]):
    root = DATA / name
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        dst = root / "images" / r.split / f"{r.stem}.jpg"
        if not dst.exists():
            os.symlink(r.image, dst)
        lines = []
        for line in open(r.label).read().splitlines():
            v = line.split()
            if len(v) < 7:
                continue
            c = int(v[0])
            if mapping is not None:
                c = mapping[c]
            lines.append(" ".join([str(c)] + v[1:]))
        (root / "labels" / r.split / f"{r.stem}.txt").write_text("\n".join(lines) + "\n")
    cfg = {"path": str(root), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": dict(enumerate(names))}
    (root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def stats_and_figures(df: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    summ = {}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        summ[s] = {
            "images": int(len(d)), "patients": int(d.patient.nunique()),
            "images_with_caries": int(d.has_caries.sum()),
            "female_frac": float((d.sex == "F").mean()),
            "age_mean": float(d.age.mean()),
            "instances": {n: int(d[f"n_{i}"].sum()) for i, n in enumerate(NAMES9)},
        }
    save_json(summ, RESULTS / "dataset_summary.json")

    rows = []
    for s in ("train", "val", "test"):
        d = summ[s]
        rows.append({"split": s, "patients": d["patients"], "images": d["images"],
                     "images w/ caries": d["images_with_caries"],
                     "female %": round(100 * d["female_frac"], 1),
                     "mean age": round(d["age_mean"], 1),
                     **{n: d["instances"][n] for n in NAMES9}})
    pd.DataFrame(rows).to_csv(RESULTS / "dataset_summary.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tot = [int(df[f"n_{i}"].sum()) for i in range(9)]
    colors = ["#8da0cb"] * 3 + ["#e15759"] * 6
    ax[0].barh(NAMES9[::-1], tot[::-1], color=colors[::-1])
    ax[0].set_xscale("log"); ax[0].set_title("Instances per class (log)")
    for i, v in enumerate(tot[::-1]):
        ax[0].text(v, i, f" {v}", va="center", fontsize=8)
    ca = df[df.n_caries > 0]
    ax[1].hist(df.age.values, bins=20, color="#4e79a7")
    ax[1].set_title("Patient age distribution (images)"); ax[1].set_xlabel("age")
    areas = []
    for r in ca.itertuples():
        for c, p in read_yolo_seg(__import__("pathlib").Path(r.label)):
            if c in CARIES_IDS9:
                areas.append(polygon_area(p) * 100)
    ax[2].hist(np.log10(np.array(areas) + 1e-6), bins=40, color="#e15759")
    ax[2].set_title("Caries lesion size (log10 % of image)")
    ax[2].set_xlabel("log10(area %)")
    fig.tight_layout(); fig.savefig(FIG / "dataset_overview.png", dpi=150); plt.close(fig)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    df = collect()
    df = split_patients(df, a.seed)
    df.to_csv(DATA / "meta.csv", index=False)
    write_yolo(df, "yolo9", None, NAMES9)
    write_yolo(df, "yolo4", MAP9TO4, NAMES4)
    s = stats_and_figures(df)
    for k, v in s.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "instances"})
    print("image sizes:", df.groupby(["width", "height"]).size().sort_values().tail(5).to_dict())


if __name__ == "__main__":
    main()
