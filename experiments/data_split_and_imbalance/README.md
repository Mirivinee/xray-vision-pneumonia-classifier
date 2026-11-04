# Start training after model handoff

The original KaggleHub images stay outside Git. The project currently has no partner baseline model, so the scripts stop with a clear message until you configure one.

## 1. Install and locate the dataset

Run in Windows PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r .\requirements.txt
& $python .\data.py
$datasetRoot = Read-Host "Paste the dataset folder printed by data.py"
```

The dataset root must directly contain `train`, `val`, and `test`, each with `NORMAL` and `PNEUMONIA`. KaggleHub selects a local cache per machine. If KaggleHub cannot resolve the dataset, provide the actual root of an existing local copy; do not guess a cache path.

On this machine, 32 redundant exact files have already been moved to a reversible quarantine. To restore them, use the generated manifest:

```powershell
$quarantineManifest = Join-Path (Split-Path $datasetRoot -Parent) "chest_xray_duplicate_quarantine\quarantine_manifest.csv"
& $python .\data\audit_duplicates.py --restore-manifest $quarantineManifest
```

The current `data/duplicate_audit.csv` is the post-quarantine report. It retains pHash candidates as informational rows; no visual review is required to create manifests. pHash similarity is not proof of duplication.

## 2. Preview and approve the split

Preview the fixed-seed, per-class stratified 80/10/10 split. Exact SHA-256 matches stay in one split. Preview mode writes no manifest:

```powershell
& $python .\experiments\data_split_and_imbalance\split_data.py --dataset-dir $datasetRoot
```

Review the counts and approve the proposal with the team. Then apply to create relative-path manifests; type `CREATE` at the prompt. Existing manifests are never overwritten and source images are not moved:

```powershell
& $python .\experiments\data_split_and_imbalance\split_data.py --dataset-dir $datasetRoot --apply
```

Outputs: `experiments/data_split_and_imbalance/manifests/splits.csv` and `summary.json`. Test rows are not loaded during training or validation.

## 3. Add the partner model

Put the partner model module somewhere importable, then set `PARTNER_MODEL_FACTORY` in `model_integration.py` or pass `--model-factory package.module:build_model`.

The loader sends RGB `float32` batches shaped `[B, 3, 224, 224]` with values in `[0, 1]`; it does not apply mean/std normalization. Labels are `NORMAL=0`, `PNEUMONIA=1`. The model factory must accept `num_classes=2` and return a model that emits `[B, 2]` logits in that class order. Training runs a dummy-batch shape/finite-output check. Confirm the model’s preprocessing expectations from its source or documentation before starting; adapt explicitly if they differ.

## 4. Train experiments separately

Replace the model factory below with the real partner import path. Approach 1 selects one imbalance mode at a time and calculates its weights from training rows only:

```powershell
$manifest = ".\experiments\data_split_and_imbalance\manifests\splits.csv"
$modelFactory = "models.baseline:build_model" # Example only; replace with actual module:function
& $python .\experiments\data_split_and_imbalance\train_approach1.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory --strategy class-weighted-loss
& $python .\experiments\data_split_and_imbalance\train_approach1.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory --strategy weighted-sampling
```

Run those as separate experiments. Approach 2 applies a random rotation of up to 3 degrees to training images only; validation remains deterministic and no augmented files are saved. Use `--rotation-degrees 1` to reduce the range (allowed: 0–10 degrees).

```powershell
& $python .\experiments\data_split_and_imbalance\train_approach2.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory
```

Each run creates a new folder under `experiments/data_split_and_imbalance/runs/` with `history.csv`, `best.pt` (selected by validation F1), and `last.pt`. Validation reports accuracy, precision, recall, F1, and confusion matrix. Test data is not used for training/model selection.

## 5. Evaluate test only after selection

After choosing a model and strategy using validation only, replace `<run-id>` with the run folder name:

```powershell
& $python .\experiments\data_split_and_imbalance\evaluate_test.py --manifest $manifest --dataset-dir $datasetRoot --checkpoint ".\experiments\data_split_and_imbalance\runs\approach1\<run-id>\best.pt" --after-selection
```

The required `--after-selection` flag is an explicit reminder. Test output is for final reporting, not further model or threshold selection.

## Checks

```powershell
& $python -m unittest discover -s .\experiments\data_split_and_imbalance\tests -v
& $python -m compileall -q .\experiments\data_split_and_imbalance
```

Training and test evaluation require a PyTorch/torchvision environment that can import its native libraries. No model training result is claimed until the partner model is available and a run is executed.
