"""Backward-compatible command-line entry point for dataset download.

New code should import from :mod:`data_pipeline`. This wrapper preserves the
existing ``python data.py`` command for teammates who already use it.
"""

from data_pipeline import (
    CLASS_TO_LABEL,
    DATASET_ID,
    SPLITS,
    ChestXrayDataset,
    create_data_loaders,
    download_dataset,
    expected_folder_errors,
    resolve_dataset_root,
)

__all__ = [
    "CLASS_TO_LABEL",
    "DATASET_ID",
    "SPLITS",
    "ChestXrayDataset",
    "create_data_loaders",
    "download_dataset",
    "expected_folder_errors",
    "resolve_dataset_root",
]


if __name__ == "__main__":
    download_dataset()
