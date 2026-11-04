"""Backward-compatible root entry point for downloading the dataset.

The implementation now lives in ``data/data_pipeline.py``. This small wrapper
preserves the existing ``python data.py`` command for teammates.
"""

from data.data_pipeline import (
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
