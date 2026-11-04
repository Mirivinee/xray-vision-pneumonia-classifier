"""Display transformed samples from the X-Ray Vision dataset pipeline."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from .data_pipeline import (
        CLASS_TO_LABEL,
        ChestXrayDataset,
        download_dataset,
        resolve_dataset_root,
    )
except ImportError:  # Supports ``python data/view_samples.py`` from the project root.
    from data_pipeline import CLASS_TO_LABEL, ChestXrayDataset, download_dataset, resolve_dataset_root


SPLITS = ("train", "val", "test")
DEFAULT_IMAGES_PER_CLASS = 4


def selected_samples(dataset, images_per_class):
    """Select indexed dataset samples while retaining their class and filename."""
    label_to_class = {label: class_name for class_name, label in CLASS_TO_LABEL.items()}
    by_class = {class_name: [] for class_name in CLASS_TO_LABEL}
    for index, (image_path, label) in enumerate(dataset.samples):
        class_name = label_to_class.get(label)
        if class_name is not None:
            by_class[class_name].append((index, image_path))

    selected = []
    for class_name in CLASS_TO_LABEL:
        if len(by_class[class_name]) < images_per_class:
            raise ValueError(
                f"Found only {len(by_class[class_name])} images for {class_name} in "
                f"the {dataset.split} split; need {images_per_class}."
            )
        selected.extend((class_name, index, path) for index, path in by_class[class_name][:images_per_class])
    return selected


def tensor_for_display(image_tensor):
    """Convert a loader tensor from [C, H, W] to a grayscale [H, W] array.

    The dataset pipeline returns RGB tensors. The source radiographs are
    grayscale, so their three channels carry the same values; averaging the
    converted [H, W, C] array preserves that grayscale appearance.
    """
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected a [3, H, W] tensor, found {tuple(image_tensor.shape)}")
    image_hwc = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.mean(image_hwc, axis=2)


def show_samples(dataset_root, split="train", images_per_class=DEFAULT_IMAGES_PER_CLASS):
    """Display a grid of samples after the project's preprocessing."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of: {', '.join(SPLITS)}")
    if images_per_class <= 0:
        raise ValueError("images_per_class must be a positive integer")

    dataset = ChestXrayDataset(dataset_root, split)
    selected = selected_samples(dataset, images_per_class)
    figure, axes = plt.subplots(
        2,
        images_per_class,
        figsize=(3.2 * images_per_class, 6),
        squeeze=False,
        constrained_layout=True,
    )

    for index, (class_name, sample_index, image_path) in enumerate(selected):
        row = 0 if class_name == "NORMAL" else 1
        column = index % images_per_class
        image_tensor, label = dataset[sample_index]
        expected_label = CLASS_TO_LABEL[class_name]
        if label != expected_label:
            raise ValueError(
                f"Label mismatch for {image_path}: expected {expected_label}, found {label}"
            )
        axes[row, column].imshow(tensor_for_display(image_tensor), cmap="gray", vmin=0, vmax=1)
        axes[row, column].set_title(f"{class_name} ({label})\n{image_path.name}", fontsize=9)
        axes[row, column].axis("off")

    figure.suptitle(f"Preprocessed X-ray samples: {split} (224 × 224 RGB → grayscale display)", fontsize=14)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Display preprocessed grayscale chest X-ray samples.")
    parser.add_argument("--split", choices=SPLITS, default="train", help="dataset split (default: train)")
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help="number of images from each class (default: 4)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="optional existing dataset directory; otherwise KaggleHub resolves it",
    )
    args = parser.parse_args()

    try:
        if args.dataset_dir is not None:
            dataset_root = resolve_dataset_root(args.dataset_dir)
            print(f"Using supplied dataset path: {dataset_root}")
        else:
            dataset_root = download_dataset()

        print(
            f"Displaying {args.images_per_class} preprocessed NORMAL and "
            f"{args.images_per_class} preprocessed PNEUMONIA images from {args.split}."
        )
        show_samples(dataset_root, args.split, args.images_per_class)
    except (FileNotFoundError, ImportError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
