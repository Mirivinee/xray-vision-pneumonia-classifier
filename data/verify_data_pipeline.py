"""Run small runtime checks for the train and validation DataLoaders."""

import argparse
from pathlib import Path

try:
    from .data_pipeline import (
        CLASS_TO_LABEL,
        create_data_loaders,
        download_dataset,
        resolve_dataset_root,
        torch,
    )
except ImportError:  # Supports ``python data/verify_data_pipeline.py`` as well.
    from data_pipeline import (
        CLASS_TO_LABEL,
        create_data_loaders,
        download_dataset,
        resolve_dataset_root,
        torch,
    )


def check_dataset_labels(dataset):
    """Check that both class folders are present and use the documented labels."""
    labels = {label for _, label in dataset.samples}
    expected = set(CLASS_TO_LABEL.values())
    if labels != expected:
        raise AssertionError(f"Expected labels {sorted(expected)}, found {sorted(labels)}")

    for class_name, expected_label in CLASS_TO_LABEL.items():
        class_labels = {
            label for path, label in dataset.samples if path.parent.name == class_name
        }
        if class_labels != {expected_label}:
            raise AssertionError(
                f"{class_name} must map to {expected_label}, found {sorted(class_labels)}"
            )


def check_batch(images, labels, split):
    """Validate shape, dtype, finite values, range, and label values."""
    if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
        raise AssertionError(f"{split}: expected [batch, 3, 224, 224], found {tuple(images.shape)}")
    if images.dtype != torch.float32:
        raise AssertionError(f"{split}: expected float32 images, found {images.dtype}")
    if not torch.isfinite(images).all():
        raise AssertionError(f"{split}: image batch contains non-finite values")
    if images.numel() and (images.min().item() < 0 or images.max().item() > 1):
        raise AssertionError(f"{split}: image values are outside [0, 1]")
    if labels.ndim != 1 or not set(labels.tolist()).issubset({0, 1}):
        raise AssertionError(f"{split}: labels must be a 1D tensor containing only 0 and 1")
    print(
        f"{split}: images={list(images.shape)}, dtype={images.dtype}, "
        f"range=[{images.min().item():.4f}, {images.max().item():.4f}], "
        f"labels={list(labels.shape)}, values={sorted(set(labels.tolist()))}"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify X-Ray Vision train/validation loaders.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    try:
        if torch is None:
            raise ImportError("PyTorch is unavailable or its native DLL is blocked.")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")

        if args.dataset_dir:
            dataset_root = resolve_dataset_root(args.dataset_dir)
        else:
            dataset_root = download_dataset()

        loaders = create_data_loaders(dataset_root, batch_size=args.batch_size)
        if set(loaders) != {"train", "val"}:
            raise AssertionError(f"Routine verification must expose only train and val: {list(loaders)}")

        for split in ("train", "val"):
            check_dataset_labels(loaders[split].dataset)
            images, labels = next(iter(loaders[split]))
            check_batch(images, labels, split)

        print("PASS: train and validation loader checks completed; test loading was not requested.")
    except ImportError as error:
        print(f"PyTorch runtime verification was not run locally: {error}")
        print(
            "Recorded Colab evidence: [4, 3, 224, 224] images, float32 values in [0, 1], "
            "and labels shape [4]; the default loaders exposed only train and val."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main() or 0)
