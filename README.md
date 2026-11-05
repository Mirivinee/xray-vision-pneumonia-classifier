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

## Shared preparation pipeline

The combined pipeline is implemented in `data/preparation_pipeline.py`; `experiments/data_split_and_imbalance/split_data.py` remains a compatibility command. It checks readability, proposes same-class SHA-256 duplicates, retains report-only average-hash candidates, and on explicit apply quarantines exact copies before creating the shared stratified manifest. RGB and grayscale PNGs are generated under `data/processed/` from that same manifest and share stable image IDs and split assignments. Source images are left in place. Optional image dimensions are handled separately in `data/dimension_handler.py`.

Generated manifests and image variants are local and ignored by Git. Before training on a fresh clone, run the confirmed pipeline apply command below to create the manifest and variants for your own dataset.

From PowerShell, run a dry report first:

```powershell
$python = ".\.venv\Scripts\python.exe"
$datasetRoot = Read-Host "Enter dataset root containing train, val, and test"
& $python .\data\preparation_pipeline.py --dataset-dir $datasetRoot --overwrite-report
```

Review `data/processed/data_report.txt`. To quarantine exact duplicates, refresh the existing shared manifest, and generate both image variants at 224×224, run:

```powershell
& $python .\data\preparation_pipeline.py --dataset-dir $datasetRoot --apply --overwrite-manifests --overwrite-report
```

Confirm the overwrite and quarantine prompts. To preserve dimensions, add `--no-resize`; configure another size with `--image-size WIDTH HEIGHT`. Existing RGB/grayscale outputs are never overwritten; choose a new `--output-dir` or review/remove an old generated output through your own data-management procedure before another apply run.

The default batch contract is `[batch, 3, 224, 224]` with `float32` image tensors and a `[batch]` integer-label tensor. `image_size` remains configurable. The merged scratch CNN in `cnn_scratch/scratch_cnn.py` accepts this format and returns one raw pneumonia logit per image; `experiments/data_split_and_imbalance/` contains manifest-based experiment runners. The experiment handoff uses `BCEWithLogitsLoss` and the same `NORMAL = 0`, `PNEUMONIA = 1` labels. The model documents no channel normalization; the training integration therefore uses the existing RGB resize and `[0, 1]` conversion. Transfer-learning compatibility has not been checked because no transfer-learning model code is present.

For fair comparisons on the existing fixed manifest, use the three experiment commands in [the experiment README](experiments/data_split_and_imbalance/README.md). They keep test data out of training and validation-based selection. Confirm with the model owner that this scratch CNN is the intended baseline, that the documented input scaling is still expected, and whether the default mild rotation for Approach 2 is acceptable.

## Grad-CAM and visual review

Validation-only Grad-CAM tools and the artefact-review checklist are in [experiments/explainability/README.md](experiments/explainability/README.md). The runner uses the saved scratch CNN checkpoint and the existing split manifest; it does not offer the official test split for visualization.

## Day 2 handoff

Run `data\verify_data_pipeline.py` to check train and validation batches, finite values, range, channel count, labels, class presence, and the `NORMAL=0` / `PNEUMONIA=1` mapping. It does not request or iterate over the test split. On this Windows environment, PyTorch's native DLL is blocked by Application Control; the recorded Colab check produced `[4, 3, 224, 224]` `float32` batches with values in `[0, 1]` and labels shape `[4]`. Test loading remains opt-in and was not part of that Colab check.

## Verified Day 1 findings and limitations

- Original-folder inspection recorded 5,824 images: 1,579 `NORMAL` and 4,245 `PNEUMONIA`, across the source folders with counts listed in [data/dataset_report.md](data/dataset_report.md).
- The current shared 80/10/10 manifest assigns: train 1,263 `NORMAL`/3,396 `PNEUMONIA`; validation 158/425; test 158/424. These are manifest partitions, not the original folder counts.
- The latest pipeline dry run found 5,824 readable images, 0 unreadable images, and 0 exact duplicate groups. Its average-hash screen found 263 candidate groups; candidates are not confirmed duplicates and are never cleaned automatically.
- Exact-byte checks do not establish patient-level separation and cannot detect visually identical files saved with different encodings.
- No source dataset images or existing split manifests were changed during the pipeline dry run.

## Data safety

Do not copy or commit X-ray images, the KaggleHub cache, credentials, virtual environments, checkpoints, or other generated artifacts. [.gitignore](.gitignore) excludes common local data and generated-file paths.
