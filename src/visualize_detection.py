"""Qualitative figure: ground-truth caries vs. predicted caries masks on
held-out test patients, for the 9-class and the unified 4-class detector."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import CARIES_IDS9, DATA, FIG, ROOT, read_yolo_seg


def main(n=4, seed=3):
    meta = pd.read_csv(DATA / "meta.csv")
    te = meta[(meta.split == "test") & (meta.n_caries >= 2)].drop_duplicates("patient").sample(n, random_state=seed)
    models = {"9-class": (YOLO(str(ROOT / "weights/yolo9_yolo11n-seg_1024.pt")), CARIES_IDS9),
              "unified 4-class": (YOLO(str(ROOT / "weights/yolo4_yolo11n-seg_1024.pt")), [3])}
    fig, ax = plt.subplots(n, 3, figsize=(15, 3.4 * n))
    for i, r in enumerate(te.itertuples()):
        img = cv2.cvtColor(cv2.imread(r.image), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]
        gt = img.copy()
        for c, p in read_yolo_seg(__import__("pathlib").Path(r.label)):
            if c in CARIES_IDS9:
                cv2.polylines(gt, [np.round(p * [w, h]).astype(np.int32)], True, (0, 255, 0), 3)
        ax[i, 0].imshow(gt); ax[i, 0].set_title(f"{r.stem}: ground-truth caries" if i == 0 else r.stem, fontsize=9)
        for j, (name, (m, ids)) in enumerate(models.items()):
            res = m.predict(r.image, imgsz=1024, conf=0.25, verbose=False, retina_masks=True)[0]
            vis = img.copy()
            if res.masks is not None:
                cls = res.boxes.cls.cpu().numpy().astype(int)
                for k, poly in enumerate(res.masks.xy):
                    if cls[k] in ids and len(poly) > 2:
                        cv2.polylines(vis, [poly.astype(np.int32)], True, (255, 40, 40), 3)
            ax[i, j + 1].imshow(vis)
            if i == 0:
                ax[i, j + 1].set_title(f"{name} detector (caries, conf≥0.25)", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "detection_examples.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
