# Split, imbalance, and augmentation experiment notes

## Dataset state

The original report recorded 5,856 readable JPEGs. On 2026-11-05, a fresh SHA-256 audit found 30 exact groups and 32 redundant same-class copies. The 32 files were moved to a reversible quarantine beside the KaggleHub dataset version. The quarantine manifest stores original paths, quarantine paths, and hashes. No pHash candidate was moved. Current split counts are train 1,340 NORMAL / 3,850 PNEUMONIA, validation 8 / 8, and test 231 / 387 (5,824 images total). The official folders are still intact apart from those explicit reversible moves.

The current `data/duplicate_audit.csv` was refreshed after quarantine: 0 exact groups, 26 DCT pHash candidate groups at Hamming threshold 0, and 4 candidates spanning existing splits. pHash candidates remain informational only. A matching hash is not proof of identity, duplicate status, or patient identity. No visual review is needed to create a split.

No patient-ID or patient-group metadata was found. Patient-level separation cannot be guaranteed.

## Dataset setup and exact duplicate restore

Install the existing project dependencies and ask KaggleHub to print its resolved dataset path:

```powershell
$python = "C:\Users\<your-user>\AppData\Local\Python\pythoncore-3.12-64\python.exe"
& $python -m pip install -r .\requirements.txt
& $python .\data.py
$datasetRoot = Read-Host "Paste the dataset folder printed by data.py"
```

The root passed to the scripts must directly contain `train`, `val`, and `test`, each with `NORMAL` and `PNEUMONIA`. KaggleHub selects the cache path for each machine; the scripts do not embed a personal cache path. If KaggleHub cannot resolve/download the dataset, use a local copy only after confirming its actual root.

To restore the quarantined copies, use the manifest created during this run:

```powershell
$quarantineManifest = Join-Path (Split-Path $datasetRoot -Parent) "chest_xray_duplicate_quarantine\quarantine_manifest.csv"
& $python .\data\audit_duplicates.py --restore-manifest $quarantineManifest
```

Exact duplicate quarantine was performed separately from split creation. pHash candidates were not changed.

## Deterministic split

`split_data.py` previews a seeded (42) 80/10/10 stratified partition by class. It groups byte-identical images by SHA-256 so a group cannot cross splits. pHash candidates remain in the audit report but do not block or group the split. Preview mode writes no manifest. It checks folder presence, split totals, per-class targets, deterministic rerun consistency, and exact-group separation. It does not infer patient identity.

```powershell
& $python .\experiments\data_split_and_imbalance\split_data.py --dataset-dir $datasetRoot
```

Review the printed counts and approve the split with the team. Then create manifests by explicitly applying and typing `CREATE` at the prompt:

```powershell
& $python .\experiments\data_split_and_imbalance\split_data.py --dataset-dir $datasetRoot --apply
```

The script writes `experiments/data_split_and_imbalance/manifests/splits.csv` and `summary.json`; it refuses to overwrite existing files. Images are not copied or moved. Validation and test retain their natural class distributions. Test data is not loaded during training or model selection.

## Model handoff

Add the partner’s model module to the repository and configure [model_integration.py](model_integration.py), or pass `--model-factory module.path:function`. The factory must accept `num_classes=2` and return a PyTorch module. If no model is configured, the scripts stop with a handoff message.

Required contract:

- Input: RGB, `float32`, `[B, 3, 224, 224]`, values `[0, 1]`.
- Preprocessing: RGB conversion, deterministic 224×224 resize and tensor scaling; no mean/std normalization.
- Output: `[B, 2]` logits ordered `NORMAL=0`, `PNEUMONIA=1`.

Training performs a dummy-batch shape/finite-logit check. It cannot infer whether the model expects different normalization or channel handling; compare its source/docs and adapt explicitly before training.

## Approach 1: imbalance handling

Select exactly one strategy. Loss and sample weights are calculated from training rows only; validation and test are not rebalanced.

```powershell
$manifest = ".\experiments\data_split_and_imbalance\manifests\splits.csv"
$modelFactory = "models.baseline:build_model" # Replace with the partner's actual import path
& $python .\experiments\data_split_and_imbalance\train_approach1.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory --strategy class-weighted-loss
& $python .\experiments\data_split_and_imbalance\train_approach1.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory --strategy weighted-sampling
```

The two commands are separate experiments. Do not combine the strategies.

## Approach 2: augmentation

Augmentation is dynamic and training-only: a random rotation up to 3 degrees by default, configurable from 0 to 10 degrees. No flip or saved augmented copies are used. Validation transforms are deterministic, and the training helper does not load test rows. Confirm the rotation with the model owners.

```powershell
& $python .\experiments\data_split_and_imbalance\train_approach2.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory
# Optional milder setting:
& $python .\experiments\data_split_and_imbalance\train_approach2.py --manifest $manifest --dataset-dir $datasetRoot --model-factory $modelFactory --rotation-degrees 1
```

Each run saves `history.csv`, `best.pt` (best validation F1), and `last.pt` under `experiments/data_split_and_imbalance/runs/`. Existing run folders are not overwritten. Validation history includes accuracy, precision, recall, F1, and a confusion matrix with rows=true and columns=predicted in NORMAL/PNEUMONIA order.

## Held-out test evaluation

Complete model/strategy selection using validation data first. Only then evaluate one selected checkpoint:

```powershell
& $python .\experiments\data_split_and_imbalance\evaluate_test.py --manifest $manifest --dataset-dir $datasetRoot --checkpoint ".\experiments\data_split_and_imbalance\runs\approach1\<run-id>\best.pt" --after-selection
```

The command requires the explicit `--after-selection` acknowledgment and prints accuracy, precision, recall, F1, and confusion matrix. Do not use test results to select models, strategies, thresholds, or hyperparameters.

## Remaining checks

1. Run the split preview and inspect actual post-quarantine counts before approving manifests.
2. Obtain the baseline module and its normalization/output documentation; run the model-contract check.
3. Confirm whether the mild rotation is acceptable.
4. Keep patient-level leakage as a stated limitation unless trustworthy patient IDs become available.
5. Run the standard-library tests with `& $python -m unittest discover -s .\experiments\data_split_and_imbalance\tests -v`; run training only in a working PyTorch/torchvision environment.
