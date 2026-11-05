"""Manifest loaders and shared training/evaluation helpers for both approaches."""

import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_scratch.train_scratch_cnn import compute_metrics, run_epoch


IMAGE_SIZE = (224, 224)
LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
SPLITS = ("train", "val", "test")
STRATEGIES = ("class-weighted-loss", "weighted-sampling")
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifests" / "splits.csv"


def read_manifest(manifest_path, dataset_root, split):
    """Read one partition and confirm its paths and class labels."""
    root = Path(dataset_root).resolve()
    if split not in SPLITS:
        raise ValueError(f"split must be one of: {', '.join(SPLITS)}")
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"relative_path", "label", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            if row["split"] != split:
                continue
            if row["label"] not in LABEL_TO_ID:
                raise ValueError(f"Unknown label {row['label']!r} in manifest")
            path = (root / row["relative_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Manifest path escapes dataset root: {row['relative_path']}")
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {path}")
            rows.append({"path": path, "relative_path": row["relative_path"], "label": row["label"]})
    if not rows:
        raise ValueError(f"No images found for split {split!r} in {manifest_path}")
    return rows


def class_counts(train_rows):
    """Count labels using training rows only and require both classes."""
    counts = Counter(row["label"] for row in train_rows)
    if set(counts) != set(LABEL_TO_ID):
        raise ValueError("Training partition must contain both NORMAL and PNEUMONIA")
    return {label: counts[label] for label in LABEL_TO_ID}


def class_pos_weight(train_rows):
    """BCEWithLogitsLoss positive-class weight, computed from training rows only."""
    counts = class_counts(train_rows)
    return counts["NORMAL"] / counts["PNEUMONIA"]


def sample_weights(train_rows):
    """Inverse-frequency weights for a training-only WeightedRandomSampler."""
    counts = class_counts(train_rows)
    return [1.0 / counts[row["label"]] for row in train_rows]


class ManifestChestXrayDataset(Dataset):
    """Load one manifest partition with RGB resize/tensor transforms."""

    def __init__(self, rows, split, augment=False, rotation_degrees=3):
        if not 0 <= rotation_degrees <= 10:
            raise ValueError("rotation_degrees must be between 0 and 10")
        self.rows = rows
        self.split = split
        self.augmented = bool(augment and split == "train" and rotation_degrees > 0)
        operations = []
        if self.augmented:
            operations.append(transforms.RandomRotation(degrees=rotation_degrees))
        operations.extend((
            transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ))
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row["path"]) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as error:
            raise OSError(f"Could not read image {row['path']}: {error}") from error
        return tensor, LABEL_TO_ID[row["label"]]


def make_training_loaders(manifest_path, dataset_root, batch_size=32, strategy=None,
                          augment_train=False, rotation_degrees=3, num_workers=0):
    """Return train and natural validation loaders; this function never loads test."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    if strategy not in (None, *STRATEGIES):
        raise ValueError(f"strategy must be one of: {', '.join(STRATEGIES)}")
    train_rows = read_manifest(manifest_path, dataset_root, "train")
    val_rows = read_manifest(manifest_path, dataset_root, "val")
    class_counts(train_rows)
    train_dataset = ManifestChestXrayDataset(
        train_rows, "train", augment=augment_train, rotation_degrees=rotation_degrees
    )
    val_dataset = ManifestChestXrayDataset(
        val_rows, "val", augment=augment_train, rotation_degrees=rotation_degrees
    )

    sampler = None
    if strategy == "weighted-sampling":
        sampler = WeightedRandomSampler(sample_weights(train_rows), len(train_rows), replacement=True)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=(sampler is None), sampler=sampler, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, train_rows, val_rows


def make_test_loader(manifest_path, dataset_root, batch_size=32, num_workers=0):
    """Build a deterministic test loader only for the separate final evaluator."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    rows = read_manifest(manifest_path, dataset_root, "test")
    dataset = ManifestChestXrayDataset(rows, "test", augment=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers), rows


def compute_training_pixel_statistics(train_rows, batch_size=32, num_workers=0):
    """Calculate RGB mean/std from the unaugmented training partition only.

    These statistics are recorded for reproducibility but are not applied: the
    teammate's CNN expects the existing [0,1] tensor preprocessing.
    """
    dataset = ManifestChestXrayDataset(train_rows, "train", augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_square_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in loader:
        images = images.to(dtype=torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_square_sum += images.square().sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
    mean = channel_sum / pixel_count
    variance = (channel_square_sum / pixel_count - mean.square()).clamp_min(0)
    return {"mean_rgb": mean.tolist(), "std_rgb": variance.sqrt().tolist(), "pixel_count": pixel_count}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directory(output_dir, approach):
    """Create a unique output folder; never overwrite an existing run."""
    if output_dir is not None:
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"Run output already exists: {destination}")
        destination.mkdir(parents=True)
        return destination
    parent = Path(__file__).resolve().parent / "runs" / approach
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir()
    return destination


def write_history(path, history):
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_training(args, approach, strategy=None, augment_train=False):
    """Run shared training and validation while keeping test rows out of memory."""
    if args.epochs <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("epochs/learning_rate must be positive and weight_decay nonnegative")
    seed_everything(args.seed)
    loaders, train_rows, val_rows = make_training_loaders(
        args.manifest, args.dataset_dir, args.batch_size, strategy=strategy,
        augment_train=augment_train, rotation_degrees=getattr(args, "rotation_degrees", 0),
        num_workers=args.num_workers,
    )
    counts = class_counts(train_rows)
    statistics = compute_training_pixel_statistics(train_rows, args.batch_size, args.num_workers)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    from model_integration import build_model
    model = build_model(dropout=args.dropout).to(device)
    sample_images, _ = next(iter(loaders["train"]))
    with torch.no_grad():
        sample_output = model(sample_images[:2].to(device))
    if tuple(sample_output.shape) != (min(2, len(sample_images)),):
        raise ValueError(f"ScratchCNN must return one logit per image; got {tuple(sample_output.shape)}")

    pos_weight_value = class_pos_weight(train_rows) if strategy == "class-weighted-loss" else None
    pos_weight = (torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                  if pos_weight_value is not None else None)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    output_dir = create_run_directory(args.output_dir, approach)
    history_path = output_dir / "history.json"
    history = []
    best_val_loss = float("inf")
    config = {
        "approach": approach,
        "strategy": strategy or "none",
        "model": "cnn_scratch.ScratchCNN",
        "dropout": args.dropout,
        "input_shape": [3, 224, 224],
        "output_shape_per_batch": "[B] one raw PNEUMONIA logit",
        "label_to_id": LABEL_TO_ID,
        "preprocessing": "RGB, bilinear resize 224x224, ToTensor [0,1], no channel normalization",
        "training_pixel_statistics_unaugmented": statistics,
        "train_class_counts": counts,
        "pos_weight": pos_weight_value,
        "augmentation": {"training_only": augment_train,
                          "rotation_degrees": getattr(args, "rotation_degrees", 0) if augment_train else 0},
        "seed": args.seed,
        "manifest": str(Path(args.manifest).resolve()),
        "dataset_root": str(Path(args.dataset_dir).resolve()),
    }
    print(f"Train rows: {len(train_rows)} {counts}; validation rows: {len(val_rows)}")
    print(f"Training-only pixel stats (recorded, not applied): {statistics}")
    print(f"Approach: {approach}; strategy: {strategy or 'none'}; device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_result = run_epoch(model, loaders["val"], criterion, device)
        record = {"epoch": epoch, "train": train_result, "val": val_result}
        history.append(record)
        write_history(history_path, history)
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "val_metrics": val_result,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
        print(f"Epoch {epoch}/{args.epochs}: train loss={train_result['loss']:.4f}; "
              f"val loss={val_result['loss']:.4f}, acc={val_result['accuracy']:.3f}, "
              f"recall={val_result['recall']:.3f}, F1={val_result['f1']:.3f}")
    print(f"Saved {output_dir / 'best.pt'}, {output_dir / 'last.pt'}, and {history_path}")
    print("Test rows were not loaded or evaluated.")
    return output_dir


def add_train_arguments(parser):
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="dataset root containing paths referenced by the manifest")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="new run directory; existing directories are never overwritten")
