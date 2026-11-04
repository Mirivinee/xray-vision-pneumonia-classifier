# X-Ray Vision data pipeline

Day 1 utilities for classifying pediatric chest X-rays as `NORMAL` or `PNEUMONIA` with PyTorch. The source images stay in KaggleHub's local cache; this repository contains code and documentation only.

## Start here for data

Open [data_pipeline.py](data_pipeline.py) first. It resolves the KaggleHub dataset root, validates the existing folders, defines `ChestXrayDataset`, and creates train/validation loaders with test loading opt-in. [inspect_data.py](inspect_data.py) is the read-only inspection/reporting utility. [dataset_report.md](dataset_report.md) contains the verified Day 1 findings. [data.py](data.py) is a small compatibility entry point for the existing download command.

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
& $python -m pip install kagglehub pillow numpy torch torchvision
& $python .\data.py
& $python .\inspect_data.py
```

The interpreter path above is an example. Use the Python interpreter where the packages are installed. PyTorch installation can vary by CPU/GPU and Python version; use the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) if the simple install does not provide a working build.

To open a few readable images from both classes:

```powershell
& $python .\inspect_data.py --show-samples
```

To inspect an already downloaded copy, pass its local directory without changing the code:

```powershell
& $python .\inspect_data.py --dataset-dir "D:\path\to\dataset"
```

Inspection checks folder structure, class counts, class balance, formats, dimensions, readable files, and exact byte duplicates. It writes the report to `dataset_report.md` and never modifies images or creates splits. Kaggle may require credentials or dataset-term acceptance before download.

## Create batches

```python
from data_pipeline import create_data_loaders, download_dataset

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
