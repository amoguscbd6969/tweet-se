import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset
import timm


RARITY_TO_AUX = {
    1: {"name_foil": 0, "art_foil": 0, "full_foil": 0, "name_silver": 0, "name_rainbow": 0, "art_sparkle": 0, "art_diagonal": 0, "art_grid": 0, "full_25th": 0, "full_clear": 0},
    2: {"name_foil": 1, "art_foil": 0, "full_foil": 0, "name_silver": 1, "name_rainbow": 0, "art_sparkle": 0, "art_diagonal": 0, "art_grid": 0, "full_25th": 0, "full_clear": 0},
    3: {"name_foil": 0, "art_foil": 1, "full_foil": 0, "name_silver": 0, "name_rainbow": 0, "art_sparkle": 1, "art_diagonal": 0, "art_grid": 0, "full_25th": 0, "full_clear": 0},
    4: {"name_foil": 1, "art_foil": 1, "full_foil": 0, "name_silver": 1, "name_rainbow": 0, "art_sparkle": 1, "art_diagonal": 0, "art_grid": 0, "full_25th": 0, "full_clear": 0},
    5: {"name_foil": 1, "art_foil": 1, "full_foil": 0, "name_silver": 0, "name_rainbow": 1, "art_sparkle": 0, "art_diagonal": 1, "art_grid": 0, "full_25th": 0, "full_clear": 0},
    6: {"name_foil": 1, "art_foil": 1, "full_foil": 0, "name_silver": 0, "name_rainbow": 1, "art_sparkle": 0, "art_diagonal": 0, "art_grid": 1, "full_25th": 0, "full_clear": 0},
    7: {"name_foil": 1, "art_foil": 1, "full_foil": 1, "name_silver": 0, "name_rainbow": 1, "art_sparkle": 0, "art_diagonal": 0, "art_grid": 1, "full_25th": 1, "full_clear": 0},
    8: {"name_foil": 1, "art_foil": 1, "full_foil": 1, "name_silver": 0, "name_rainbow": 1, "art_sparkle": 0, "art_diagonal": 0, "art_grid": 1, "full_25th": 0, "full_clear": 1},
}

AUX_TARGETS = [
    "name_foil", "art_foil", "full_foil",  # official task 2/3/4
    "name_silver", "name_rainbow",           # 2 extra for name
    "art_sparkle", "art_diagonal", "art_grid", "art_foil",  # 4 cues for art (art_foil is shared)
    "full_25th", "full_clear",               # 2 cues for full-card foil
]

# Which crop each binary model uses.
TASK_TO_VIEW = {
    "name_foil": "name",
    "name_silver": "name",
    "name_rainbow": "name",
    "art_foil": "art",
    "art_sparkle": "art",
    "art_diagonal": "art",
    "art_grid": "art",
    "full_foil": "full",
    "full_25th": "full",
    "full_clear": "full",
}

# 11 binary models = 8 rarity cues + official 3 tasks.
MODEL_TASKS = [
    "name_silver", "name_rainbow",
    "art_sparkle", "art_diagonal", "art_grid", "art_foil",
    "full_25th", "full_clear",
    "name_foil", "full_foil", "art_foil",
]


@dataclass
class TrainConfig:
    train_csv: str
    image_dir: str
    out_dir: str = "outputs"
    image_size: int = 448
    batch_size: int = 16
    epochs: int = 6
    folds: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    swa_start: int = 4
    tta: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enrich_labels(df: pd.DataFrame) -> pd.DataFrame:
    for rarity, labels in RARITY_TO_AUX.items():
        mask = df["rarity"] == rarity
        for key, value in labels.items():
            df.loc[mask, key] = value
    for c in set().union(*[d.keys() for d in RARITY_TO_AUX.values()]):
        df[c] = df[c].astype(int)
    return df


def get_crop(img: np.ndarray, view: str) -> np.ndarray:
    h, w = img.shape[:2]
    if view == "name":
        y1, y2 = int(0.06 * h), int(0.17 * h)
        x1, x2 = int(0.08 * w), int(0.92 * w)
        return img[y1:y2, x1:x2]
    if view == "art":
        y1, y2 = int(0.22 * h), int(0.64 * h)
        x1, x2 = int(0.11 * w), int(0.89 * w)
        return img[y1:y2, x1:x2]
    return img


def build_transforms(size: int) -> Dict[str, A.Compose]:
    name_tf = A.Compose([
        A.Resize(size, size),
        A.ColorJitter(brightness=0.25, contrast=0.35, saturation=0.45, hue=0.03, p=0.9),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])
    art_tf = A.Compose([
        A.Resize(size, size),
        A.ColorJitter(brightness=0.15, contrast=0.25, saturation=0.35, hue=0.02, p=0.8),
        A.RandomGamma(gamma_limit=(80, 130), p=0.5),
        A.Sharpen(alpha=(0.1, 0.35), p=0.4),
        A.Normalize(),
        ToTensorV2(),
    ])
    full_tf = A.Compose([
        A.Resize(size, size),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.25, hue=0.02, p=0.7),
        A.RandomToneCurve(scale=0.08, p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_tf = A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])
    return {"name": name_tf, "art": art_tf, "full": full_tf, "val": val_tf}


class CardDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, task: str, train: bool, tfm: Dict[str, A.Compose]):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.task = task
        self.view = TASK_TO_VIEW[task]
        self.train = train
        self.transforms = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.image_dir / row["id"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = get_crop(img, self.view)
        tf = self.transforms[self.view] if self.train else self.transforms["val"]
        tensor = tf(image=img)["image"]
        target = torch.tensor(row[self.task], dtype=torch.float32)
        return tensor, target


class BinaryBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=True, num_classes=1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_one_task(cfg: TrainConfig, df: pd.DataFrame, task: str) -> Dict:
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    transforms = build_transforms(cfg.image_size)
    device = torch.device(cfg.device)

    fold_states = []
    oof_pred = np.zeros(len(df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df["rarity"])):
        tr_df = df.iloc[tr_idx]
        va_df = df.iloc[va_idx]
        tr_ds = CardDataset(tr_df, cfg.image_dir, task, True, transforms)
        va_ds = CardDataset(va_df, cfg.image_dir, task, False, transforms)
        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        model_r18 = BinaryBackbone("resnet18").to(device)
        model_b4 = BinaryBackbone("tf_efficientnet_b4").to(device)
        models = [model_r18, model_b4]

        opts = [torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in models]
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=cfg.epochs) for o in opts]
        swa_models = [AveragedModel(m).to(device) for m in models]
        swa_scheds = [SWALR(o, swa_lr=cfg.lr * 0.2) for o in opts]
        criterion = nn.BCEWithLogitsLoss()

        best_fold, best_f1 = None, -1.0
        for epoch in range(cfg.epochs):
            for m in models:
                m.train()
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                for m, o in zip(models, opts):
                    o.zero_grad()
                    logits = m(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    o.step()

            if epoch >= cfg.swa_start:
                for sm, m in zip(swa_models, models):
                    sm.update_parameters(m)
                for ss in swa_scheds:
                    ss.step()
            else:
                for s in scheds:
                    s.step()

            pred = infer_ensemble(models, va_loader, device)
            score = f1_score(va_df[task].values, (pred >= 0.5).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_fold = [copy.deepcopy(m.state_dict()) for m in models]

        if cfg.epochs > cfg.swa_start:
            for sm, loader in zip(swa_models, [tr_loader, tr_loader]):
                update_bn(loader, sm, device=device)

        for m, st in zip(models, best_fold):
            m.load_state_dict(st)

        pred = infer_ensemble(models, va_loader, device)
        oof_pred[va_idx] = pred
        fold_states.append({"resnet18": best_fold[0], "effb4": best_fold[1]})
        print(f"[{task}] fold={fold} f1={f1_score(va_df[task].values, (pred >= 0.5).astype(int), zero_division=0):.4f}")

    return {
        "oof": oof_pred,
        "states": fold_states,
        "f1": f1_score(df[task].values, (oof_pred >= 0.5).astype(int), zero_division=0),
    }


@torch.no_grad()
def infer_ensemble(models: List[nn.Module], loader: DataLoader, device: torch.device, tta: int = 1) -> np.ndarray:
    preds = []
    for m in models:
        m.eval()

    for x, _ in loader:
        x = x.to(device)
        p = 0
        for _ in range(tta):
            out = 0
            for m in models:
                out += torch.sigmoid(m(x))
            out = out / len(models)
            p += out
        p = p / tta
        preds.append(p.detach().cpu().numpy())
    return np.concatenate(preds)


def decode_rarity_from_binary(features: Dict[str, int]) -> int:
    for rarity, target in RARITY_TO_AUX.items():
        ok = True
        for k in ["name_silver", "name_rainbow", "art_sparkle", "art_diagonal", "art_grid", "full_25th", "full_clear"]:
            if features.get(k, 0) != target[k]:
                ok = False
                break
        if ok:
            return rarity
    # fallback by official foil triplet
    key = (features["name_foil"], features["art_foil"], features["full_foil"])
    table = {
        (0, 0, 0): 1,
        (1, 0, 0): 2,
        (0, 1, 0): 3,
        (1, 1, 0): 4,
        (1, 1, 1): 7,
    }
    return table.get(key, 4)


def post_process(row: Dict[str, int]) -> Dict[str, int]:
    rarity = row["rarity"]
    trio = (row["name_foil"], row["art_foil"], row["full_foil"])
    expected = (RARITY_TO_AUX[rarity]["name_foil"], RARITY_TO_AUX[rarity]["art_foil"], RARITY_TO_AUX[rarity]["full_foil"])
    if trio != expected:
        candidates = []
        for r, vals in RARITY_TO_AUX.items():
            if trio == (vals["name_foil"], vals["art_foil"], vals["full_foil"]):
                candidates.append(r)
        if len(candidates) == 1:
            row["rarity"] = candidates[0]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    cfg = TrainConfig(train_csv=args.train_csv, image_dir=args.image_dir, out_dir=args.out_dir)
    seed_all(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    df = pd.read_csv(cfg.train_csv)
    df = enrich_labels(df)

    tasks = ["name_silver", "name_rainbow", "art_sparkle", "art_diagonal", "art_grid", "full_25th", "full_clear", "name_foil", "art_foil", "full_foil"]
    all_results = {}
    for task in tasks:
        result = train_one_task(cfg, df, task)
        all_results[task] = {"f1": result["f1"]}
        df[f"pred_{task}"] = (result["oof"] >= 0.5).astype(int)
        torch.save(result["states"], Path(cfg.out_dir) / f"{task}_ensemble.pt")

    df["rarity_pred"] = df.apply(lambda r: decode_rarity_from_binary({
        "name_silver": int(r["pred_name_silver"]),
        "name_rainbow": int(r["pred_name_rainbow"]),
        "art_sparkle": int(r["pred_art_sparkle"]),
        "art_diagonal": int(r["pred_art_diagonal"]),
        "art_grid": int(r["pred_art_grid"]),
        "full_25th": int(r["pred_full_25th"]),
        "full_clear": int(r["pred_full_clear"]),
        "name_foil": int(r["pred_name_foil"]),
        "art_foil": int(r["pred_art_foil"]),
        "full_foil": int(r["pred_full_foil"]),
    }), axis=1)

    refined = []
    for _, r in df.iterrows():
        row = {
            "rarity": int(r["rarity_pred"]),
            "name_foil": int(r["pred_name_foil"]),
            "art_foil": int(r["pred_art_foil"]),
            "full_foil": int(r["pred_full_foil"]),
        }
        refined.append(post_process(row))
    pred_df = pd.DataFrame(refined)

    rarity_f1 = f1_score(df["rarity"], pred_df["rarity"], average="macro")
    name_f1 = f1_score(df["name_foil"], pred_df["name_foil"])
    art_f1 = f1_score(df["art_foil"], pred_df["art_foil"])
    full_f1 = f1_score(df["full_foil"], pred_df["full_foil"])
    score = 0.8 * rarity_f1 + 0.05 * name_f1 + 0.1 * art_f1 + 0.05 * full_f1

    metrics = {
        "rarity_f1": rarity_f1,
        "name_foil_f1": name_f1,
        "art_foil_f1": art_f1,
        "full_foil_f1": full_f1,
        "weighted_score": score,
        "per_task": all_results,
    }
    with open(Path(cfg.out_dir) / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
