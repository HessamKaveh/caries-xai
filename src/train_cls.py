"""Step 5 - Train crop-level caries screening classifiers.

--split patient : the project's patient-level split (default, leakage-free)
--split random  : crops shuffled at random regardless of patient. Used ONLY for
                  the leakage experiment, to quantify how much a naive split
                  inflates reported performance.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from common import DATA, RESULTS, ROOT, save_json, seed_all

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch: str) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(arch)
    return m


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    return model.layer4[-1] if arch == "resnet50" else model.features[-1]


class Crops(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
               T.ColorJitter(0.15, 0.15, 0.1, 0.02)] if train else []
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = Image.open(DATA / "crops" / "img" / f"{r.crop}.jpg").convert("RGB")
        return self.tf(x), torch.tensor(float(r.y))


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.to(dev)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--split", default="patient", choices=["patient", "random"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    dev = "cuda"
    df = pd.read_csv(DATA / "crops" / "crops.csv")
    if a.split == "random":
        rng = np.random.default_rng(a.seed)
        df["split"] = rng.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.1, 0.2])
    tag = f"{a.arch}_{a.split}"
    if (RESULTS / f"cls_train_{tag}.json").exists():
        print("SKIP (already trained)", tag); return
    dls = {s: DataLoader(Crops(df[df.split == s], s == "train"), batch_size=a.bs,
                         shuffle=s == "train", num_workers=6, pin_memory=True)
           for s in ("train", "val", "test")}
    model = build_model(a.arch).to(dev)
    pos = (df[df.split == "train"].y == 1).mean()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dls["train"]),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best, hist, start = -1, [], 0
    ck = ROOT / "weights" / f"cls_{tag}.pt"; ck.parent.mkdir(exist_ok=True)
    state_f = ROOT / "weights" / f"cls_{tag}_last.pt"
    if state_f.exists():  # resume after interruption
        st = torch.load(state_f, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        best, hist, start = st["best"], st["hist"], st["epoch"] + 1
        torch.set_rng_state(st["rng"])
        print(f"[{tag}] RESUME from epoch {start}", flush=True)
    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); tl = 0
        for x, y in dls["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(x)
        lv = predict(model, dls["val"], dev)
        auc = roc_auc_score(dls["val"].dataset.df.y, lv)
        hist.append({"epoch": ep, "train_loss": tl / len(dls["train"].dataset), "val_auroc": auc})
        print(f"[{tag}] ep {ep} loss {hist[-1]['train_loss']:.4f} val AUROC {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best:
            best = auc; torch.save(model.state_dict(), ck)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist,
                    "rng": torch.get_rng_state()}, state_f)
    model.load_state_dict(torch.load(ck))
    out = {}
    for s in ("val", "test"):
        d = dls[s].dataset.df.copy(); d["logit"] = predict(model, dls[s], dev)
        d.to_csv(RESULTS / f"cls_logits_{tag}_{s}.csv", index=False)
        out[s] = roc_auc_score(d.y, d.logit)
    save_json({"arch": a.arch, "split": a.split, "best_val_auroc": best, "test_auroc": out["test"],
               "history": hist, "n": {s: len(dls[s].dataset) for s in dls}},
              RESULTS / f"cls_train_{tag}.json")
    state_f.unlink(missing_ok=True)
    print("TEST AUROC", tag, out["test"])


if __name__ == "__main__":
    main()
