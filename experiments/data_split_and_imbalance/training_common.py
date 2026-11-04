"""Shared manifest datasets, train/validation loaders, and metrics."""

import argparse
import csv
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_SIZE = (224, 224)
STRATEGIES = ("class-weighted-loss", "weighted-sampling")


def require_torch():
    """Import training libraries only when a training/evaluation command runs."""
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision import transforms
    except Exception as error:
        raise RuntimeError(
            "Training/evaluation needs working PyTorch, torchvision, and Pillow. "
            "If Windows blocks a PyTorch DLL, use an approved PyTorch-enabled environment."
        ) from error
    return torch, Image, DataLoader, WeightedRandomSampler, transforms


def load_manifest_rows(manifest_path, dataset_root, split):
    """Read and validate just one manifest split; paths are relative to dataset_root."""
    root = Path(dataset_root).resolve()
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"relative_path", "label", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest must contain: {', '.join(sorted(required))}")
        for row in reader:
            if row["split"] != split:
                continue
            if row["label"] not in LABEL_TO_ID:
                raise ValueError(f"Unknown class label: {row['label']}")
            image_path = (root / row["relative_path"]).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"Manifest path escapes dataset root: {row['relative_path']}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            rows.append({"path": image_path, "label": row["label"], "split": split})
    if not rows:
        raise ValueError(f"No images found in manifest for split {split!r}.")
    return rows


def compute_class_weights(train_rows):
    """Return inverse-frequency loss weights from training rows only."""
    counts = Counter(row["label"] for row in train_rows)
    if set(counts) != set(LABEL_TO_ID):
        raise ValueError("The training split must contain both NORMAL and PNEUMONIA.")
    total = len(train_rows)
    return {label: total / (len(LABEL_TO_ID) * counts[label]) for label in LABEL_TO_ID}


def compute_sample_weights(train_rows):
    """Return per-image sampling weights from training rows only."""
    counts = Counter(row["label"] for row in train_rows)
    if set(counts) != set(LABEL_TO_ID):
        raise ValueError("The training split must contain both NORMAL and PNEUMONIA.")
    return [1.0 / counts[row["label"]] for row in train_rows]


def validate_strategy(strategy):
    """Reject combinations or names outside the two mutually exclusive modes."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Choose exactly one strategy: {', '.join(STRATEGIES)}")


def augmentation_enabled(split, augment_train):
    """Augmentation is enabled for the training split only."""
    return split == "train" and augment_train


def validate_rotation_degrees(degrees):
    """Keep this X-ray rotation setting mild and configurable."""
    if not 0 <= degrees <= 10:
        raise ValueError("rotation-degrees must be between 0 and 10")


def make_dataset_class(torch, Image, transforms):
    """Create a Dataset using RGB, deterministic 224x224 resize, and [0,1] tensors."""
    class ManifestDataset(torch.utils.data.Dataset):
        def __init__(self, rows, split, augment_train=False, rotation_degrees=3):
            self.rows = rows
            validate_rotation_degrees(rotation_degrees)
            operations = []
            if augmentation_enabled(split, augment_train) and rotation_degrees:
                operations.append(transforms.RandomRotation(degrees=rotation_degrees))
            operations.extend((transforms.Resize(IMAGE_SIZE), transforms.ToTensor()))
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

    return ManifestDataset


def make_split_loader(manifest_path, dataset_root, split, batch_size, augment_train=False,
                      rotation_degrees=3, sampler=None):
    """Load exactly one requested split; the test evaluator opts into test explicitly."""
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    torch, Image, DataLoader, _, transforms = require_torch()
    rows = load_manifest_rows(manifest_path, dataset_root, split)
    Dataset = make_dataset_class(torch, Image, transforms)
    dataset = Dataset(rows, split, augment_train=augment_train, rotation_degrees=rotation_degrees)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train" and sampler is None), sampler=sampler)
    return torch, loader, rows


def make_train_val_loaders(manifest_path, dataset_root, batch_size, strategy="none",
                           augment_train=False, rotation_degrees=3):
    """Make train and natural validation loaders; never loads test data."""
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if strategy != "none":
        validate_strategy(strategy)
    torch, Image, DataLoader, WeightedRandomSampler, transforms = require_torch()
    Dataset = make_dataset_class(torch, Image, transforms)
    train_rows = load_manifest_rows(manifest_path, dataset_root, "train")
    val_rows = load_manifest_rows(manifest_path, dataset_root, "val")
    train_set = Dataset(train_rows, "train", augment_train, rotation_degrees)
    val_set = Dataset(val_rows, "val", augment_train, rotation_degrees)

    sampler = None
    if strategy == "weighted-sampling":
        weights = compute_sample_weights(train_rows)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
    }
    return torch, loaders, train_rows


def metrics_from_counts(tn, fp, fn, tp):
    """Calculate binary metrics with PNEUMONIA as the positive class."""
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tn + tp) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate(model, loader, torch, device, criterion):
    """Return average loss and accuracy/precision/recall/F1/confusion matrix."""
    model.eval()
    total_loss = 0.0
    tn = fp = fn = tp = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            tn += int(((predictions == 0) & (labels == 0)).sum().item())
            fp += int(((predictions == 1) & (labels == 0)).sum().item())
            fn += int(((predictions == 0) & (labels == 1)).sum().item())
            tp += int(((predictions == 1) & (labels == 1)).sum().item())
    result = metrics_from_counts(tn, fp, fn, tp)
    result["loss"] = total_loss / (tn + fp + fn + tp)
    return result


def validate_model_output(model, torch, device):
    """Check the model accepts RGB batches and returns two class logits."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(torch.zeros(2, 3, *IMAGE_SIZE, dtype=torch.float32, device=device))
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (2, 2):
            shape = tuple(output.shape) if hasattr(output, "shape") else type(output).__name__
            raise ValueError(f"Model must return a [B, 2] logits tensor; got {shape} for B=2.")
        if not torch.isfinite(output).all():
            raise ValueError("Model returned non-finite logits for the contract smoke check.")
    except Exception as error:
        raise RuntimeError(
            "Model contract check failed. Expected RGB float32 [B,3,224,224] values in [0,1] "
            "and [B,2] logits ordered NORMAL=0, PNEUMONIA=1. Adapt the model explicitly if needed."
        ) from error
    finally:
        model.train(was_training)


def add_training_arguments(parser):
    parser.add_argument("--manifest", type=Path, required=True, help="approved splits.csv")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="dataset root used by the manifest")
    parser.add_argument("--model-factory", help="module.path:function; defaults to model_integration.py")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, help="new run folder; must not already exist")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda")


def create_run_dir(output_dir, approach):
    """Create a fresh run folder; default names use UTC and never overwrite."""
    if output_dir:
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"Output folder already exists: {destination}")
        destination.mkdir(parents=True)
        return destination
    runs = Path(__file__).resolve().parent / "runs" / approach
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if destination.exists():
        raise FileExistsError(f"Run folder already exists: {destination}; pass a fresh --output-dir")
    destination.mkdir()
    return destination


def train(args, approach, strategy="none", augment_train=False, rotation_degrees=3):
    """Train one run, validate each epoch, and save history plus best/last checkpoints."""
    if args.batch_size <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("batch-size, epochs, and learning-rate must be positive")
    validate_rotation_degrees(rotation_degrees)

    from model_integration import build_partner_model, resolve_factory_spec

    factory_spec = resolve_factory_spec(args.model_factory)
    torch, loaders, train_rows = make_train_val_loaders(
        args.manifest, args.dataset_dir, args.batch_size, strategy,
        augment_train=augment_train, rotation_degrees=rotation_degrees,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_partner_model(factory_spec)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = model.to(device)
    validate_model_output(model, torch, device)

    criterion = torch.nn.CrossEntropyLoss()
    if strategy == "class-weighted-loss":
        weights = compute_class_weights(train_rows)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(
            [weights["NORMAL"], weights["PNEUMONIA"]], dtype=torch.float32, device=device
        ))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run_dir = create_run_dir(args.output_dir, approach)
    history_path = run_dir / "history.csv"
    best_f1 = -1.0
    metadata = {
        "approach": approach,
        "strategy": strategy,
        "model_factory": factory_spec,
        "image_size": list(IMAGE_SIZE),
        "label_to_id": LABEL_TO_ID,
        "augmentation": {"training_only": augment_train, "rotation_degrees": rotation_degrees if augment_train else 0},
        "seed": args.seed,
    }
    fields = ["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    print(f"Output folder: {run_dir}")
    print(f"Strategy: {strategy}; augmentation: {'training only' if augment_train else 'off'}")
    with history_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = total = 0
            for images, labels in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
            validation = evaluate(model, loaders["val"], torch, device, criterion)
            tn, fp = validation["confusion_matrix"][0]
            fn, tp = validation["confusion_matrix"][1]
            row = {"epoch": epoch, "train_loss": total_loss / total, "val_loss": validation["loss"],
                   "accuracy": validation["accuracy"], "precision": validation["precision"],
                   "recall": validation["recall"], "f1": validation["f1"], "tn": tn, "fp": fp, "fn": fn, "tp": tp}
            writer.writerow(row)
            output.flush()
            checkpoint = {**metadata, "epoch": epoch, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(), "validation": validation}
            torch.save(checkpoint, run_dir / "last.pt")
            if validation["f1"] >= best_f1:
                best_f1 = validation["f1"]
                torch.save(checkpoint, run_dir / "best.pt")
            print(f"Epoch {epoch}/{args.epochs}: accuracy={validation['accuracy']:.4f}, "
                  f"precision={validation['precision']:.4f}, recall={validation['recall']:.4f}, "
                  f"F1={validation['f1']:.4f}, confusion={validation['confusion_matrix']}")
    print(f"History: {history_path}")
    print(f"Best validation checkpoint: {run_dir / 'best.pt'}")
    print("Test data was not loaded or evaluated.")
    return run_dir
