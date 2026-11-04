"""Step 6 - Quantitative explainability audit.

Saliency methods are scored against expert caries masks (plausibility) and
by perturbation (faithfulness). Two trivial reference maps are included so a
method has to beat "always look at the centre" and random noise to count.

Methods : Grad-CAM, Grad-CAM++, Integrated Gradients (blurred baseline),
          Occlusion (blurred patches), Centre prior, Random.
Metrics : pointing game (8 px tolerance), relevance mass in lesion,
          relevance lift (mass / lesion area), pixel AUROC vs mask,
          deletion AUC (lower = more faithful), insertion AUC (higher = better).
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import DATA, FIG, RESULTS, ROOT, save_json, seed_all
from train_cls import MEAN, STD, build_model, target_layer

S = 224
DEV = "cuda"
MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def to_tensor(img: np.ndarray) -> torch.Tensor:  # HWC uint8 RGB -> normalised NCHW
    x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255
    return (x - MEAN_T) / STD_T


def blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), 12)


def norm(s: np.ndarray) -> np.ndarray:
    s = np.maximum(s, 0)
    return s / (s.max() + 1e-12)


class CAM:
    def __init__(self, model, layer):
        self.model = model; self.a = self.g = None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.a = o
        if o.requires_grad:
            o.register_hook(lambda g: setattr(self, "g", g))

    def __call__(self, x, plus=False):
        self.model.zero_grad()
        x = x.to(DEV).requires_grad_(True)
        self.model(x)[0, 0].backward()
        A, G = self.a.detach()[0], self.g.detach()[0]
        if plus:
            g2, g3 = G ** 2, G ** 3
            den = 2 * g2 + A.sum((1, 2), keepdim=True) * g3
            alpha = torch.where(G != 0, g2 / (den + 1e-7), torch.zeros_like(G))
            w = (alpha * F.relu(G)).sum((1, 2))
        else:
            w = G.mean((1, 2))
        cam = F.relu((w[:, None, None] * A).sum(0))
        cam = F.interpolate(cam[None, None], (S, S), mode="bilinear", align_corners=False)[0, 0]
        return norm(cam.cpu().numpy())


def integrated_gradients(model, img, steps=32):
    x = to_tensor(img); b = to_tensor(blur(img))
    alphas = torch.linspace(0, 1, steps + 1)[1:].view(-1, 1, 1, 1)
    grads = torch.zeros_like(x)
    for chunk in alphas.split(16):
        xi = (b + chunk * (x - b)).to(DEV).requires_grad_(True)
        model(xi)[:, 0].sum().backward()
        grads += xi.grad.detach().cpu().sum(0, keepdim=True)
    attr = ((x - b) * grads / steps)[0].sum(0).numpy()
    attr = cv2.GaussianBlur(np.maximum(attr, 0), (0, 0), 3)
    return norm(attr)


@torch.no_grad()
def prob(model, imgs: list[np.ndarray]) -> np.ndarray:
    x = torch.cat([to_tensor(i) for i in imgs]).to(DEV)
    out = []
    for c in x.split(64):
        out.append(torch.sigmoid(model(c)[:, 0]).cpu())
    return torch.cat(out).numpy()


def occlusion(model, img, patch=32, stride=16):
    bl = blur(img); p0 = prob(model, [img])[0]
    sal = np.zeros((S, S)); cnt = np.zeros((S, S)); imgs, boxes = [], []
    for y in range(0, S - patch + 1, stride):
        for x in range(0, S - patch + 1, stride):
            im = img.copy(); im[y:y + patch, x:x + patch] = bl[y:y + patch, x:x + patch]
            imgs.append(im); boxes.append((y, x))
    ps = prob(model, imgs)
    for (y, x), p in zip(boxes, ps):
        sal[y:y + patch, x:x + patch] += p0 - p; cnt[y:y + patch, x:x + patch] += 1
    return norm(sal / np.maximum(cnt, 1))


CENTRE = norm(np.exp(-(((np.arange(S) - S / 2)[None] ** 2 + (np.arange(S) - S / 2)[:, None] ** 2) / (2 * 56 ** 2))))


def deletion_insertion(model, img, sal, steps=20):
    bl = blur(img); order = np.argsort(-sal.ravel())
    dels, ins = [], []
    for k in np.linspace(0, 1, steps + 1):
        m = np.zeros(S * S, bool); m[order[: int(k * S * S)]] = True; m = m.reshape(S, S, 1)
        dels.append(np.where(m, bl, img)); ins.append(np.where(m, img, bl))
    p = prob(model, dels + ins)
    d, i = p[: steps + 1], p[steps + 1:]
    return float(np.trapezoid(d, dx=1 / steps)), float(np.trapezoid(i, dx=1 / steps))


def plaus(sal, mask):
    dil = cv2.dilate(mask, np.ones((17, 17), np.uint8))
    yx = np.unravel_index(np.argmax(sal), sal.shape)
    mass = float((sal * mask).sum() / (sal.sum() + 1e-12))
    return {"pointing": float(dil[yx] > 0), "mass": mass, "lift": mass / max(mask.mean(), 1e-6),
            "pix_auroc": float(roc_auc_score(mask.ravel(), sal.ravel() + 1e-9 * np.random.rand(S * S)))}


def run(arch: str, n: int, seed: int):
    seed_all(seed)
    model = build_model(arch).to(DEV)
    model.load_state_dict(torch.load(ROOT / "weights" / f"cls_{arch}_patient.pt"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cam = CAM(model, target_layer(model, arch))
    lg = pd.read_csv(RESULTS / f"cls_logits_{arch}_patient_test.csv")
    pos = lg[(lg.y == 1) & (lg.mask_frac > 0.002)]
    pos = pos.sample(min(n, len(pos)), random_state=seed)
    rng = np.random.default_rng(seed)
    rows, examples = [], []
    for r in tqdm(pos.itertuples(), total=len(pos), desc=arch):
        img = np.array(Image.open(DATA / "crops" / "img" / f"{r.crop}.jpg").convert("RGB"))
        mask = (cv2.imread(str(DATA / "crops" / "mask" / f"{r.crop}.png"), 0) > 127).astype(np.uint8)
        x = to_tensor(img)
        sals = {
            "Grad-CAM": cam(x), "Grad-CAM++": cam(x, plus=True),
            "Integrated Gradients": integrated_gradients(model, img),
            "Occlusion": occlusion(model, img),
            "Centre prior": CENTRE, "Random": norm(cv2.GaussianBlur(rng.random((S, S)), (0, 0), 8)),
        }
        for mth, s in sals.items():
            d, i = deletion_insertion(model, img, s)
            rows.append({"arch": arch, "crop": r.crop, "method": mth, "correct": int(r.logit > 0),
                         "deletion_auc": d, "insertion_auc": i, **plaus(s, mask)})
        if len(examples) < 6 and r.logit > 0:
            examples.append((img, mask, sals))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"xai_per_crop_{arch}.csv", index=False)
    figure(examples, arch)
    return df


def figure(examples, arch):
    meths = ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Occlusion"]
    fig, ax = plt.subplots(len(examples), 1 + len(meths), figsize=(2.3 * (1 + len(meths)), 2.3 * len(examples)))
    for i, (img, mask, sals) in enumerate(examples):
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im = cv2.drawContours(img.copy(), cnt, -1, (0, 255, 0), 2)
        ax[i, 0].imshow(im); ax[i, 0].set_title("crop + caries GT" if i == 0 else "", fontsize=9)
        for j, m in enumerate(meths):
            ax[i, j + 1].imshow(img); ax[i, j + 1].imshow(sals[m], cmap="jet", alpha=0.45)
            ax[i, j + 1].contour(mask, levels=[0.5], colors="white", linewidths=0.8)
            if i == 0:
                ax[i, j + 1].set_title(m, fontsize=9)
    for a_ in ax.ravel():
        a_.axis("off")
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"xai_examples_{arch}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["resnet50", "efficientnet_b0"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    allr = pd.concat([run(arch, a.n, a.seed) for arch in a.archs])
    summ = allr.groupby(["arch", "method"])[["pointing", "mass", "lift", "pix_auroc", "deletion_auc",
                                             "insertion_auc"]].mean().round(4)
    summ.to_csv(RESULTS / "xai_summary.csv")
    save_json({f"{k[0]}|{k[1]}": v for k, v in summ.to_dict(orient="index").items()},
              RESULTS / "xai_summary.json")
    print(summ)


if __name__ == "__main__":
    main()
