# Training with the existing stratified manifest

## Rebuild manifests safely after exact-duplicate cleanup

`data/preparation_pipeline.py` is now the shared implementation. `split_data.py` remains as the old command path and delegates to it. The pipeline checks readability, hashes raw bytes using SHA-256 within the same class, keeps the lexicographically first relative path, and retains average-hash candidates as report-only. The dry run writes `data/processed/data_report.txt` plus JSON/CSV details and does not change images or manifests.

```powershell
$python = ".\.venv\Scripts\python.exe"
$datasetRoot = Read-Host "Enter dataset root containing train, val, and test"
& $python .\data\preparation_pipeline.py --dataset-dir $datasetRoot
```

Review `data/processed/data_report.txt`, `preparation_report.json`, and `exact_duplicates.csv`. To quarantine exact redundant copies, refresh the deterministic, class-stratified 80/10/10 manifest (seed 42), and write RGB/grayscale variants, use `--apply`. This project already has manifests and a dry-run report, so add `--overwrite-manifests --overwrite-report` and confirm the prompts. Quarantined files are placed under `<dataset-root>/.duplicate_quarantine/` and recorded in its `manifest.json` for restoration.

```powershell
& $python .\data\preparation_pipeline.py --dataset-dir $datasetRoot --apply --overwrite-manifests --overwrite-report
```

To restore all quarantined copies, after confirming the original paths are free:

```powershell
& $python .\data\preparation_pipeline.py --dataset-dir $datasetRoot --restore
```

Restoring files does not rewrite manifests; rerun the dry run and then the confirmed apply command if you want manifests to include restored files again.

The manifest columns remain `relative_path,label,split`; the summary retains the seed, ratios, per-class counts, and patient-separation limitation. The script refuses to replace existing manifests without the explicit overwrite flag and interactive confirmation.

The model adapter connects to `cnn_scratch.ScratchCNN`. The manifest at `manifests/splits.csv` is reused; these scripts do not create or change splits or images.

## Interface and preprocessing

The teammate CNN accepts RGB `float32` images `[B,3,224,224]` scaled to `[0,1]`, and returns one raw PNEUMONIA logit per image `[B]`. Its labels are `NORMAL=0`, `PNEUMONIA=1`; training uses `BCEWithLogitsLoss`. The loader converts to RGB, bilinearly resizes to 224×224, and applies `ToTensor`. It does not apply ImageNet mean/std normalization, matching the CNN’s documented input.

Each run calculates per-channel mean/std from unaugmented training rows only and saves those statistics in the checkpoint metadata. They are recorded, not applied, so model input preprocessing remains unchanged.

## Commands (Windows PowerShell, from the repository root)

Set the dataset root to the actual folder that contains `train`, `val`, and `test`; the split CSV paths are relative to this root:

```powershell
$python = ".\.venv\Scripts\python.exe"
$datasetRoot = Read-Host "Paste the dataset root"
$manifest = ".\experiments\data_split_and_imbalance\manifests\splits.csv"
```

Approach 1 runs exactly one imbalance strategy per command. Loss weighting uses `NORMAL_count / PNEUMONIA_count` from training rows; weighted sampling also uses training rows only. Validation/test distributions remain natural.

```powershell
& $python .\experiments\data_split_and_imbalance\train_approach1.py --dataset-dir $datasetRoot --manifest $manifest --strategy class-weighted-loss
& $python .\experiments\data_split_and_imbalance\train_approach1.py --dataset-dir $datasetRoot --manifest $manifest --strategy weighted-sampling
```

Approach 2 uses the same CNN, manifest, validation metrics, and checkpoint-selection rule. It applies random rotation up to 3 degrees on training images only. Validation remains deterministic; no augmented images are saved.

```powershell
& $python .\experiments\data_split_and_imbalance\train_approach2.py --dataset-dir $datasetRoot --manifest $manifest
# Optional range adjustment, limited to 0–10 degrees:
& $python .\experiments\data_split_and_imbalance\train_approach2.py --dataset-dir $datasetRoot --manifest $manifest --rotation-degrees 1
```

Runs write a new folder under `runs/` with `best.pt`, `last.pt`, and `history.json`; validation loss selects `best.pt`. Existing run folders are not overwritten. The test split is never loaded during training or model selection.

After choosing the model using validation data, evaluate the selected checkpoint on test once:

```powershell
& $python .\experiments\data_split_and_imbalance\evaluate_test.py --dataset-dir $datasetRoot --manifest $manifest --checkpoint ".\experiments\data_split_and_imbalance\runs\approach1\<run-id>\best.pt" --after-selection
```

The test evaluator reports the same accuracy, precision, recall, specificity, F1, and AUC setup as the teammate’s baseline script. The threshold remains 0.5; do not tune it on test data.

## Small smoke test and tests

The smoke test runs one batch and one optimizer step for class-weighted loss, weighted sampling, and training-only augmentation. It does not save a checkpoint or run full training.

```powershell
& $python .\experiments\data_split_and_imbalance\smoke_test.py --dataset-dir $datasetRoot --manifest $manifest
& $python -m unittest discover -s .\experiments\data_split_and_imbalance\tests -v
```
