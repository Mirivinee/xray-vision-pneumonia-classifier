"""Generate a balanced, validation-only Grad-CAM review set and CSV."""

import argparse
import csv
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_split_and_imbalance.training_common import (
    DEFAULT_MANIFEST,
    LABEL_TO_ID,
    ManifestChestXrayDataset,
    read_manifest,
)
from experiments.explainability.grad_cam import compute_grad_cam
from experiments.explainability.model_adapters import (
    ARCHITECTURES,
    load_model_checkpoint,
    pneumonia_probability,
    resolve_normalization,
)
from experiments.explainability.visualize import save_contact_sheet, save_overlay


def choose_review_indices(predictions, per_class, seed):
    """Select each true class evenly, including correct and incorrect examples when available."""
    rng = random.Random(seed)
    selected = []
    for label in ("NORMAL", "PNEUMONIA"):
        candidates = [row for row in predictions if row["true_label"] == label]
        correct = [row for row in candidates if row["true_label"] == row["predicted_label"]]
        incorrect = [row for row in candidates if row["true_label"] != row["predicted_label"]]
        rng.shuffle(correct)
        rng.shuffle(incorrect)
        chosen = []
        if per_class >= 2 and correct and incorrect:
            chosen.extend((correct.pop(), incorrect.pop()))
        remaining = correct + incorrect
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, per_class - len(chosen))])
        selected.extend(chosen)
    return sorted(selected, key=lambda row: (row["true_label"], row["index"]))


def predict_validation(model, loader, device):
    """Score all validation rows so the review sample can include errors when present."""
    rows = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = pneumonia_probability(logits).cpu().tolist()
            labels = labels.tolist()
            for label, probability in zip(labels, probabilities):
                predicted = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
                true_label = "PNEUMONIA" if label == LABEL_TO_ID["PNEUMONIA"] else "NORMAL"
                rows.append({
                    "index": offset,
                    "true_label": true_label,
                    "predicted_label": predicted,
                    "pneumonia_score": float(probability),
                    "confidence": max(float(probability), 1.0 - float(probability)),
                })
                offset += 1
    return rows


def create_output_dir(requested):
    if requested is not None:
        destination = Path(requested)
        if destination.exists():
            raise FileExistsError(f"Output directory already exists: {destination}")
        destination.mkdir(parents=True)
        return destination
    destination = Path(__file__).resolve().parent / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalization", choices=("auto", "none", "imagenet"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        parser.error("--samples-per-class and --batch-size must be positive")
    try:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
        model, target_layer, checkpoint = load_model_checkpoint(args.architecture, args.checkpoint, device)
        normalization = resolve_normalization(args.normalization, checkpoint)
        rows = read_manifest(args.manifest, args.dataset_dir, "val")
        dataset = ManifestChestXrayDataset(rows, "val", augment=False)
        if normalization == "imagenet":
            dataset.transform = transforms.Compose((
                dataset.transform,
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        predictions = predict_validation(model, loader, device)
        selected = choose_review_indices(predictions, args.samples_per_class, args.seed)
        output_dir = create_output_dir(args.output_dir)
        records = []
        for prediction in selected:
            index = prediction["index"]
            row = rows[index]
            image, _ = dataset[index]
            record = {
                "image_path": str(row["path"]),
                "true_label": prediction["true_label"],
                "predicted_label": prediction["predicted_label"],
                "pneumonia_score": prediction["pneumonia_score"],
                "confidence": prediction["confidence"],
                "target_class": "PNEUMONIA score",
                "target_layer": next(name for name, module in model.named_modules() if module is target_layer),
                "overlay_path": "",
                "error_status": "ok",
            }
            try:
                image, _ = dataset[index]
                cam, _ = compute_grad_cam(model, image.unsqueeze(0).to(device), target_layer)
                safe_id = f"{index:05d}_{Path(row['path']).stem}"
                overlay_path = output_dir / "overlays" / f"{safe_id}.png"
                save_overlay(image, cam[0], overlay_path)
                record["overlay_path"] = str(overlay_path)
            except Exception as error:
                record["error_status"] = f"{type(error).__name__}: {error}"
            records.append(record)

        fieldnames = ("image_path", "true_label", "predicted_label", "pneumonia_score", "confidence",
                      "target_class", "target_layer", "overlay_path", "error_status")
        csv_path = output_dir / "review.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        sheet_path = save_contact_sheet(records, output_dir / "contact_sheet.png")
        print(f"Validation images scored: {len(predictions)}; selected for review: {len(selected)}")
        print(f"Pneumonia-score Grad-CAM rows: {sum(record['error_status'] == 'ok' for record in records)} succeeded")
        print(f"CSV: {csv_path}")
        print(f"Contact sheet: {sheet_path or 'not created (no overlays succeeded)'}")
        print("Only the validation split was read; heatmaps require human interpretation.")
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

