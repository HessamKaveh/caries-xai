"""Step 2 - Train YOLO11-seg on the 9-class and the unified 4-class label spaces."""
from __future__ import annotations

import argparse
import shutil

from ultralytics import YOLO

from common import DATA, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["yolo9", "yolo4"], required=True)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    name = f"{a.labels}_{a.model.split('.')[0]}_{a.imgsz}"
    out = ROOT / "weights"; out.mkdir(exist_ok=True)
    final = out / f"{name}.pt"
    if final.exists():
        print("SKIP (already trained)", final); return
    run_dir = ROOT / "runs" / "seg" / name
    last = run_dir / "weights" / "last.pt"
    if last.exists():  # interrupted run -> continue from the last saved epoch
        print("RESUME", last)
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:  # training had already finished
            print("nothing to resume:", e)
        shutil.copy(run_dir / "weights" / "best.pt", final)
        print("BEST", final); return
    model = YOLO(a.model)
    model.train(
        data=str(DATA / a.labels / "data.yaml"), imgsz=a.imgsz, epochs=a.epochs,
        batch=a.batch, workers=a.workers, seed=a.seed, deterministic=False,
        project=str(ROOT / "runs" / "seg"), name=name, exist_ok=True,
        patience=25, cos_lr=True, close_mosaic=10, amp=True, plots=True,
        # intraoral photos: mirrored/rotated views are clinically plausible
        fliplr=0.5, flipud=0.0, degrees=10, scale=0.4, hsv_h=0.01,
    )
    shutil.copy(run_dir / "weights" / "best.pt", final)
    print("BEST", final)


if __name__ == "__main__":
    main()
