"""Train the baseline scratch CNN on the existing train split; validate on val.

Run from the project root (where the ``data/`` package lives):

    python train_scratch_cnn.py --epochs 15
    python train_scratch_cnn.py --dataset-dir /path/to/chest_xray --epochs 15
    python train_scratch_cnn.py --max-batches 2 --epochs 1      # quick smoke test

The test split is never loaded here.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# This script lives in cnn_scratch/, while the shared ``data/`` package is in
# the project root (one level up). Make the root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import create_data_loaders, download_dataset, resolve_dataset_root
try:
    from .scratch_cnn import ScratchCNN
except ImportError:  # Supports running this file directly as a script.
    from scratch_cnn import ScratchCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = max(len(labels), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0  # sensitivity for PNEUMONIA
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = None
    if len(set(labels.tolist())) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(labels, probs))
        except ImportError:
            pass
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss, seen = 0.0, 0
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device).float()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            all_labels.extend(labels.cpu().int().tolist())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the baseline scratch CNN.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", action="store_true",
                        help="weight PNEUMONIA by n_normal / n_pneumonia (Day 3 experiment)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit batches per epoch for a quick smoke test")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = (resolve_dataset_root(args.dataset_dir)
                    if args.dataset_dir else download_dataset())
    loaders = create_data_loaders(dataset_root, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    train_loader, val_loader = loaders["train"], loaders["val"]

    labels = [label for _, label in train_loader.dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Train: {n_neg} NORMAL, {n_pos} PNEUMONIA | Val: {len(val_loader.dataset)} images")

    pos_weight = None
    if args.class_weight:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        print(f"Using pos_weight={pos_weight.item():.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ScratchCNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best_val_loss = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
        val = run_epoch(model, val_loader, criterion, device, None, args.max_batches)
        history.append({"epoch": epoch, "train": train, "val": val})
        print(
            f"Epoch {epoch:02d}/{args.epochs} ({time.time() - start:.0f}s) | "
            f"train loss {train['loss']:.4f} acc {train['accuracy']:.3f} | "
            f"val loss {val['loss']:.4f} acc {val['accuracy']:.3f} "
            f"recall {val['recall']:.3f} spec {val['specificity']:.3f} f1 {val['f1']:.3f}"
        )
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "config": vars(args) | {"output_dir": str(args.output_dir),
                                                "dataset_dir": str(args.dataset_dir)}},
                       args.output_dir / "best.pt")

    with open(args.output_dir / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    print(f"Done. Best val loss {best_val_loss:.4f}. Saved best.pt and history.json "
          f"to {args.output_dir}")


if __name__ == "__main__":
    main()
