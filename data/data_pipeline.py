"""Dataset path resolution, preprocessing, and PyTorch loaders for X-Ray Vision.

The original KaggleHub dataset is read from its local cache. This module does
not copy, modify, or split the downloaded images.
"""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import kagglehub
except ImportError as error:
    raise ImportError(
        "KaggleHub is required to download the dataset. Install it with "
        "'python -m pip install kagglehub'."
    ) from error

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    # Keep path resolution and inspection imports usable when Windows blocks
    # PyTorch's native DLLs. Dataset/loader use reports the actionable error.
    torch = None
    DataLoader = None
    Dataset = object


DATASET_ID = "paultimothymooney/chest-xray-pneumonia"
SPLITS = ("train", "val", "test")
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def _require_torch():
    if torch is None or DataLoader is None:
        raise ImportError(
            "PyTorch could not be imported. Install it with 'python -m pip install torch'. "
            "If it is installed on Windows, check whether application-control policy "
            "blocked one of its native DLLs."
        )


def resolve_dataset_root(download_path):
    """Return the folder containing the existing train/val/test folders.

    KaggleHub may return a version directory containing a nested ``chest_xray``
    folder, so both layouts are supported without embedding a personal cache path.
    """
    downloaded = Path(download_path)
    candidates = (downloaded, downloaded / "chest_xray")
    for candidate in candidates:
        if any((candidate / split).is_dir() for split in SPLITS):
            return candidate
    return downloaded


def expected_folder_errors(dataset_root):
    """Return missing split/class folders for a useful validation message."""
    root = Path(dataset_root)
    return [
        Path(split) / class_name
        for split in SPLITS
        for class_name in CLASS_TO_LABEL
        if not (root / split / class_name).is_dir()
    ]


def download_dataset():
    """Download the dataset through KaggleHub and return its resolved root."""
    download_path = kagglehub.dataset_download(DATASET_ID)
    dataset_root = resolve_dataset_root(download_path)

    print(f"KaggleHub download path: {download_path}")
    print(f"Dataset folder being inspected: {dataset_root}")
    missing = expected_folder_errors(dataset_root)
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        print(f"Missing expected folder(s): {formatted}")
    else:
        print("Verified train/val/test folders and NORMAL/PNEUMONIA class folders.")
    return dataset_root


class ChestXrayDataset(Dataset):
    """Load one existing split as ``(image_tensor, integer_label)`` pairs.

    Images are converted to RGB, resized deterministically with bilinear
    interpolation, and scaled to [0, 1]. No random augmentation is applied.
    """

    def __init__(self, dataset_root, split, image_size=(224, 224)):
        _require_torch()
        if split not in SPLITS:
            raise ValueError("split must be one of: 'train', 'val', or 'test'.")
        if len(image_size) != 2 or any(int(size) <= 0 for size in image_size):
            raise ValueError("image_size must be a pair of positive integers.")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = tuple(int(size) for size in image_size)
        split_path = self.dataset_root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_path}")

        self.samples = []
        for class_name, label in CLASS_TO_LABEL.items():
            class_path = split_path / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(f"Missing expected class folder: {class_path}")
            paths = sorted(
                path for path in class_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend((path, label) for path in paths)
        if not self.samples:
            raise ValueError(f"No supported image files found under {split_path}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                height, width = self.image_size
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                array = np.array(image, dtype=np.float32, copy=True)
        except Exception as error:
            raise OSError(f"Could not read image '{image_path}': {error}") from error

        image_tensor = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
        return image_tensor, label


def create_data_loaders(
    dataset_root,
    batch_size=32,
    image_size=(224, 224),
    num_workers=0,
    include_test=False,
):
    """Create train/validation loaders and optionally an evaluation-only test loader.

    Test loading is opt-in so normal training and validation code cannot include
    test images accidentally. Do not use the test loader for tuning or selection.
    """
    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    datasets = {
        "train": ChestXrayDataset(dataset_root, "train", image_size),
        "val": ChestXrayDataset(dataset_root, "val", image_size),
    }
    if include_test:
        datasets["test"] = ChestXrayDataset(dataset_root, "test", image_size)

    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
