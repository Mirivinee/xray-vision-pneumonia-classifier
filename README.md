# X-Ray Vision data pipeline

Day 1 utilities for classifying pediatric chest X-rays as `NORMAL` or `PNEUMONIA` with PyTorch. The source images stay in KaggleHub's local cache; this repository contains code and documentation only.

## Start here for data

Open [data/data_pipeline.py](data/data_pipeline.py) first. It resolves the KaggleHub dataset root, validates the existing folders, defines `ChestXrayDataset`, and creates train/validation loaders with test loading opt-in. [data/inspect_data.py](data/inspect_data.py) is the read-only inspection/reporting utility. [data/verify_data_pipeline.py](data/verify_data_pipeline.py) checks real train and validation batches when PyTorch is available. [data/dataset_report.md](data/dataset_report.md) contains the verified Day 1 and Day 2 findings. The root [data.py](data.py) remains a small compatibility entry point for the existing download command.

All data-related implementation and documentation now lives in `data/`. The root `data.py` wrapper is retained only so existing `python .\data.py` commands continue to work.

The expected layout is:

```text
<KaggleHub dataset cache>/chest_xray/
├── train/NORMAL/
├── train/PNEUMONIA/
├── val/NORMAL/
├── val/PNEUMONIA/
├── test/NORMAL/
└── test/PNEUMONIA/
```

KaggleHub chooses the cache location for each machine, commonly under the user's local `.cache/kagglehub` directory. No personal username or cache path is hard-coded.

## Setup and inspection (Windows PowerShell)

From `D:\PROJECTS\X-Ray`:

```powershell
$python = "C:\Users\<your-user>\AppData\Local\Python\pythoncore-3.12-64\python.exe"
& $python -m pip install -r .\requirements.txt
& $python .\data.py
& $python .\data\inspect_data.py
& $python .\data\verify_data_pipeline.py
```

The interpreter path above is an example. Use the Python interpreter where the packages are installed. PyTorch installation can vary by CPU/GPU and Python version; use the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) if the simple install does not provide a working build.

To open a few readable images from both classes:

```powershell
& $python .\data\inspect_data.py --show-samples
& $python .\data\view_samples.py
# Equivalent when the active Python is already on PATH:
python data/view_samples.py
```

`view_samples.py` loads samples through `ChestXrayDataset`, so the displayed images use the same RGB conversion and 224 × 224 resize as the training pipeline. It displays four `train/NORMAL` and four `train/PNEUMONIA` images in a two-row grayscale grid. Each title contains the class label, numeric label, and filename. The defaults can be changed without modifying the dataset:

```powershell
& $python .\data\view_samples.py --split val --images-per-class 2
& $python .\data\view_samples.py --split test --images-per-class 4
& $python .\data\view_samples.py --dataset-dir "D:\path\to\dataset"
```

The viewer reads images directly from the KaggleHub dataset path and does not copy, move, rename, or modify them. Matplotlib is included in `requirements.txt`; install all project dependencies with `& $python -m pip install -r .\requirements.txt`.

To inspect an already downloaded copy, pass its local directory without changing the code:

```powershell
& $python .\data\inspect_data.py --dataset-dir "D:\path\to\dataset"
```

Inspection checks folder structure, class counts, class balance, formats, dimensions, readable files, and exact byte duplicates. It writes the report to `data\dataset_report.md` and never modifies images or creates splits. Kaggle may require credentials or dataset-term acceptance before download.

The inspection also runs a lightweight 8 × 8 average-hash screen for possible visual near-duplicates. Its candidates are not confirmed leakage: hashes can collide, and the screen cannot establish patient-level separation.

## Create batches

```python
from data.data_pipeline import create_data_loaders, download_dataset

dataset_root = download_dataset()
loaders = create_data_loaders(
    dataset_root,
    batch_size=32,
    image_size=(224, 224),
)

images, labels = next(iter(loaders["train"]))
print(images.shape)  # torch.Size([32, 3, 224, 224])
print(labels.shape)  # torch.Size([32])
```

The label mapping is `NORMAL = 0`, `PNEUMONIA = 1`. The default preprocessing converts every image to RGB, resizes it deterministically to 224 × 224 with bilinear interpolation, and scales the tensor to `[0, 1]`. No random augmentation is included yet. `create_data_loaders` returns `train` and `val`; add `include_test=True` only for final evaluation. Keep test data out of tuning, threshold selection, and model selection.

The default batch contract is `[batch, 3, 224, 224]` with `float32` image tensors and a `[batch]` integer-label tensor. `image_size` remains configurable. Anurag and Suraj should confirm that 224 × 224 RGB inputs, `[0, 1]` scaling, and the absence of initial random augmentation match their scratch-CNN and transfer-learning expectations before training. No model source files are currently present in this repository, so model-specific compatibility has not been tested here.

## Day 2 handoff

Run `data\verify_data_pipeline.py` to check train and validation batches, finite values, range, channel count, labels, class presence, and the `NORMAL=0` / `PNEUMONIA=1` mapping. It does not request or iterate over the test split. On this Windows environment, PyTorch's native DLL is blocked by Application Control; the recorded Colab check produced `[4, 3, 224, 224]` `float32` batches with values in `[0, 1]` and labels shape `[4]`. Test loading remains opt-in and was not part of that Colab check.

## Verified Day 1 findings and limitations

- Train: 1,341 `NORMAL`, 3,875 `PNEUMONIA`.
- Validation: 8 `NORMAL`, 8 `PNEUMONIA` (16 total; very small validation set).
- Test: 234 `NORMAL`, 390 `PNEUMONIA`.
- All 5,856 images were readable JPEGs, with 4,803 distinct dimensions.
- There were 30 byte-identical duplicate groups, all within one split; none crossed splits.
- Exact-byte checks do not establish patient-level separation and cannot detect visually identical files saved with different encodings.
- No dataset images were modified.

## Data safety

Do not copy or commit X-ray images, the KaggleHub cache, credentials, virtual environments, checkpoints, or other generated artifacts. [.gitignore](.gitignore) excludes common local data and generated-file paths.
