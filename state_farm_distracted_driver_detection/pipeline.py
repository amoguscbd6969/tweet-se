import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# -------------------------------
# 1) Skin/face-hand focused utils
# -------------------------------

def detect_skin_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Heuristic skin detector combining HSV and YCrCb thresholds."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([20, 170, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def upper_body_crop(img_rgb: np.ndarray, skin_mask: np.ndarray, min_ratio: float = 0.35) -> np.ndarray:
    """Bias crop toward upper body + skin regions (face/hands)."""
    h, w, _ = img_rgb.shape
    top_h = int(0.75 * h)
    upper = img_rgb[:top_h]
    upper_mask = skin_mask[:top_h]

    ys, xs = np.where(upper_mask > 0)
    if len(xs) > 50:
        x1, x2 = max(0, xs.min() - 25), min(w, xs.max() + 25)
        y1, y2 = max(0, ys.min() - 25), min(top_h, ys.max() + 25)
    else:
        x1, x2 = int(0.15 * w), int(0.85 * w)
        y1, y2 = 0, int(0.7 * h)

    crop = upper[y1:y2, x1:x2]

    # enforce minimum area to avoid too aggressive crops
    if crop.size == 0 or (crop.shape[0] * crop.shape[1]) < (min_ratio * h * w):
        return upper
    return crop


class SkinAwareTransform:
    def __init__(self, image_size: int = 224, train: bool = True):
        if train:
            self.global_aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(0.12, 0.12, 0.12, 0.06),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.0), value="random"),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.global_aug = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask = detect_skin_mask_bgr(img_bgr)
        crop = upper_body_crop(np.array(img), mask)
        return self.global_aug(Image.fromarray(crop))


# -------------------------------
# 2) Dataset
# -------------------------------

class StateFarmDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path,
        transform,
        num_classes: int = 10,
        test_mode: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_classes = num_classes
        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.root_dir / row["img_path"]
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img)

        if self.test_mode:
            return x, row["img"]

        y = int(row["label"])
        return x, y, row["img"]


# -------------------------------
# 3) Models
# -------------------------------

class ResNet18Head(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits, feat


class ConvNeXtTinyHead(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits, feat


# -------------------------------
# 4) Training helpers
# -------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 32
    workers: int = 4
    image_size: int = 224
    folds: int = 5
    seed: int = 42
    label_smoothing: float = 0.1


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer_scheduler(model: nn.Module, model_name: str, total_steps: int):
    if model_name == "resnet18":
        lr = 1e-3
        min_lr = 1e-4
    else:
        lr = 5e-4
        min_lr = 1e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(step):
        progress = min(step / total_steps, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr / lr) + (1 - min_lr / lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    running_loss = 0.0

    for x, y, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            logits, _ = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def infer_logits_and_embeddings(model, loader, device):
    model.eval()
    logits_list, emb_list, names = [], [], []

    for batch in tqdm(loader, leave=False):
        if len(batch) == 3:
            x, _, img_names = batch
        else:
            x, img_names = batch
        x = x.to(device)

        logits, feat = model(x)
        logits_list.append(logits.float().cpu())
        emb_list.append(feat.float().cpu())
        names.extend(list(img_names))

    logits = torch.cat(logits_list, dim=0).numpy()
    emb = torch.cat(emb_list, dim=0).numpy()
    return logits, emb, names


# -------------------------------
# 5) TTA + Ensembling + KNN graph smoothing
# -------------------------------

def tta_predict(model, dataset_df, root_dir, image_size, batch_size, workers, device, num_tta=2, test_mode=True):
    logits_accum = None
    emb_accum = None

    for t in range(num_tta):
        transform = SkinAwareTransform(image_size=image_size, train=(t > 0))
        ds = StateFarmDataset(dataset_df, root_dir, transform=transform, test_mode=test_mode)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        logits, emb, names = infer_logits_and_embeddings(model, dl, device)
        logits_accum = logits if logits_accum is None else logits_accum + logits
        emb_accum = emb if emb_accum is None else emb_accum + emb

    logits_accum /= num_tta
    emb_accum /= num_tta
    return logits_accum, emb_accum, names


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def knn_attention_smoothing(
    probs: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
    alpha: float = 0.7,
    temperature: float = 0.1,
    n_steps: int = 2,
) -> np.ndarray:
    """Graph-style message passing over cosine KNN graph (FAISS IP)."""
    emb = l2_normalize(embeddings.astype(np.float32))
    n, dim = emb.shape

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # +1 because first neighbor is usually self
    sim, idx = index.search(emb, k + 1)
    sim = sim[:, 1:]
    idx = idx[:, 1:]

    p = probs.copy()
    for _ in range(n_steps):
        neigh_probs = p[idx]  # (n, k, C)
        w = sim / max(temperature, 1e-6)
        w = np.exp(w - w.max(axis=1, keepdims=True))
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        agg = (w[..., None] * neigh_probs).sum(axis=1)
        p = alpha * p + (1 - alpha) * agg

    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return p


# -------------------------------
# 6) End-to-end runner
# -------------------------------

def run_pipeline(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train_labels.csv"
    test_csv = data_dir / "sample_submission.csv"

    train_df = pd.read_csv(train_csv)  # columns: img,label,driver_id(optional),img_path
    test_df = pd.read_csv(test_csv)    # columns: img,c0...c9 and img_path

    if "img_path" not in train_df.columns:
        train_df["img_path"] = "train/" + train_df["img"]
    if "img_path" not in test_df.columns:
        test_df["img_path"] = "test/" + test_df["img"]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_probs = np.zeros((len(train_df), args.num_classes), dtype=np.float32)
    test_probs_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df["label"])):
        if fold != args.fold and args.fold >= 0:
            continue

        print(f"\n===== Fold {fold} =====")
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        tr_tf = SkinAwareTransform(image_size=args.image_size, train=True)
        va_tf = SkinAwareTransform(image_size=args.image_size, train=False)

        tr_ds = StateFarmDataset(tr_df, data_dir, tr_tf, num_classes=args.num_classes, test_mode=False)
        va_ds = StateFarmDataset(va_df, data_dir, va_tf, num_classes=args.num_classes, test_mode=False)

        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        # model A: ResNet18
        model_a = ResNet18Head(num_classes=args.num_classes).to(device)
        total_steps_a = args.epochs_resnet * len(tr_dl)
        opt_a, sch_a = create_optimizer_scheduler(model_a, "resnet18", total_steps_a)

        # model B: ConvNeXt Tiny
        model_b = ConvNeXtTinyHead(num_classes=args.num_classes).to(device)
        total_steps_b = args.epochs_convnext * len(tr_dl)
        opt_b, sch_b = create_optimizer_scheduler(model_b, "convnext_tiny", total_steps_b)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        scaler_a, scaler_b = GradScaler(), GradScaler()

        for ep in range(args.epochs_resnet):
            loss = train_one_epoch(model_a, tr_dl, opt_a, sch_a, criterion, scaler_a, device)
            print(f"ResNet18 epoch {ep + 1}/{args.epochs_resnet} - loss: {loss:.4f}")

        for ep in range(args.epochs_convnext):
            loss = train_one_epoch(model_b, tr_dl, opt_b, sch_b, criterion, scaler_b, device)
            print(f"ConvNeXt epoch {ep + 1}/{args.epochs_convnext} - loss: {loss:.4f}")

        # validation predictions + embeddings with TTA
        va_logits_a, va_emb_a, _ = tta_predict(
            model_a, va_df, data_dir, args.image_size, args.batch_size, args.workers, device, num_tta=args.tta, test_mode=False
        )
        va_logits_b, va_emb_b, _ = tta_predict(
            model_b, va_df, data_dir, args.image_size, args.batch_size, args.workers, device, num_tta=args.tta, test_mode=False
        )

        va_probs = 0.5 * F.softmax(torch.tensor(va_logits_a), dim=1).numpy() + 0.5 * F.softmax(torch.tensor(va_logits_b), dim=1).numpy()
        va_emb = np.concatenate([va_emb_a, va_emb_b], axis=1)  # ~1280-dim
        va_probs_smooth = knn_attention_smoothing(
            va_probs, va_emb, k=args.knn_k, alpha=args.alpha, temperature=args.temperature, n_steps=args.smooth_steps
        )

        oof_probs[va_idx] = va_probs_smooth

        # test predictions
        te_logits_a, te_emb_a, te_names = tta_predict(
            model_a, test_df, data_dir, args.image_size, args.batch_size, args.workers, device, num_tta=args.tta, test_mode=True
        )
        te_logits_b, te_emb_b, _ = tta_predict(
            model_b, test_df, data_dir, args.image_size, args.batch_size, args.workers, device, num_tta=args.tta, test_mode=True
        )

        te_probs = 0.5 * F.softmax(torch.tensor(te_logits_a), dim=1).numpy() + 0.5 * F.softmax(torch.tensor(te_logits_b), dim=1).numpy()
        te_emb = np.concatenate([te_emb_a, te_emb_b], axis=1)
        te_probs_smooth = knn_attention_smoothing(
            te_probs, te_emb, k=args.knn_k, alpha=args.alpha, temperature=args.temperature, n_steps=args.smooth_steps
        )
        test_probs_folds.append(te_probs_smooth)

        # optional save checkpoints
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model_a.state_dict(), save_dir / f"resnet18_fold{fold}.pth")
            torch.save(model_b.state_dict(), save_dir / f"convnext_tiny_fold{fold}.pth")

    # outputs
    oof_pred = oof_probs.argmax(axis=1)
    oof_acc = (oof_pred == train_df["label"].values).mean()
    print(f"\nOOF Accuracy: {oof_acc:.5f}")

    sub_probs = np.mean(test_probs_folds, axis=0)
    sub = pd.DataFrame(sub_probs, columns=[f"c{i}" for i in range(args.num_classes)])
    sub.insert(0, "img", te_names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / "submission.csv", index=False)

    oof = pd.DataFrame(oof_probs, columns=[f"c{i}" for i in range(args.num_classes)])
    oof.insert(0, "img", train_df["img"].values)
    oof.insert(1, "label", train_df["label"].values)
    oof.to_csv(out_dir / "oof_probs.csv", index=False)



def parse_args():
    p = argparse.ArgumentParser(description="State Farm distracted driver full pipeline")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=-1, help="-1: train all folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-resnet", type=int, default=6)
    p.add_argument("--epochs-convnext", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--tta", type=int, default=2)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--smooth-steps", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
