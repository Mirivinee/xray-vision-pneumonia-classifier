# Grad-CAM and artefact review

This folder generates validation-only visualizations for the X-Ray Vision classifiers. Grad-CAM highlights image regions associated with a selected score; it is a visualization aid, not proof of causal reasoning, lung localization, or clinical reliability. The project is academic and is not a diagnostic tool.

## Models and input/output contracts

The repository contains `cnn_scratch.ScratchCNN`. It takes the existing pipeline input: RGB, float32 `[B,3,224,224]`, values `[0,1]`, and returns one raw pneumonia logit `[B]`. The label mapping is `NORMAL=0`, `PNEUMONIA=1`; sigmoid converts the scratch CNN output to P(PNEUMONIA).

Adapters are also provided for torchvision ResNet-50 and DenseNet-121 with a two-logit classifier in `[NORMAL,PNEUMONIA]` order. These are constructed with `weights=None`, so the tool does not download weights. No ResNet-50 or DenseNet-121 project checkpoint was found; their checkpoint compatibility is therefore not verified. Confirm model architecture, classifier shape, class order, and training preprocessing before using those adapters.

The scratch checkpoint stores `model_state` plus a `config` identifying the model, dropout, labels, and preprocessing. The loader also accepts a raw state dict or a `state_dict` wrapper. It does not write to a checkpoint. If the checkpoint does not identify its normalization, pass `--normalization none` or `--normalization imagenet` to match the model's actual training setup. `auto` only selects a normalization when the checkpoint metadata clearly states it. The project's existing manifest loader supplies RGB conversion, bilinear resize to 224×224, and tensor conversion. ImageNet normalization, if explicitly requested, is applied afterward.

Grad-CAM targets the scratch model's declared last convolution. For ResNet-50, the adapter selects the last convolution in `layer4`; for DenseNet-121, the last convolution in `features.denseblock4`. The selected layer is checked against the loaded model before generating a map. A two-logit model's pneumonia score is column 1; a one-logit model's pneumonia score is its raw output.

## Run a balanced validation review

From the repository root in PowerShell, set the local dataset root to the directory containing the paths referenced by the existing manifest:

```powershell
$python = ".\.venv\Scripts\python.exe"
$datasetRoot = Read-Host "Enter dataset root containing the split-manifest paths"
$manifest = ".\experiments\data_split_and_imbalance\manifests\splits.csv"
```

Run the available scratch CNN checkpoint. In this workspace, enter the existing local `best.pt` path when prompted:

```powershell
$checkpoint = Read-Host "Enter the local ScratchCNN best.pt path"
& $python .\experiments\explainability\run_gradcam.py `
  --architecture scratch_cnn `
  --checkpoint $checkpoint `
  --dataset-dir $datasetRoot `
  --manifest $manifest `
  --samples-per-class 8
```

The checkpoint used for the documented initial review is local under `experiments/data_split_and_imbalance/runs/` and is ignored by Git; it is not included for teammates cloning the repository. Supply a local compatible checkpoint when running the command.

The run reads and scores validation rows, then selects up to eight per true class. When both prediction outcomes are available and the requested sample size is at least two, the selection includes correct and incorrect predictions. Outputs go to a new timestamped folder under `experiments/explainability/outputs/` (or pass a new `--output-dir`). The folder contains `review.csv`, individual 224×224 overlays, and `contact_sheet.png`. Existing output folders are not overwritten.

For ResNet-50 or DenseNet-121, use the corresponding `--architecture` and a compatible binary checkpoint. Those project checkpoints were not found in this repository. Supply explicit `--normalization none` or `--normalization imagenet` if checkpoint metadata does not record it. Never select normalization based on the Grad-CAM appearance; match the model's training configuration.

```powershell
& $python .\experiments\explainability\run_gradcam.py --architecture resnet50 --checkpoint .\path\to\resnet50_checkpoint.pt --dataset-dir $datasetRoot --manifest $manifest --normalization imagenet
& $python .\experiments\explainability\run_gradcam.py --architecture densenet121 --checkpoint .\path\to\densenet121_checkpoint.pt --dataset-dir $datasetRoot --manifest $manifest --normalization imagenet
```

The `imagenet` values above are examples only; use them only if those checkpoints were trained with that normalization.

The command has no test-split option: the artifact-review workflow is validation-only. A CSV row records image path, true/predicted label, P(PNEUMONIA), predicted-class confidence, target score/layer, overlay path, and any per-image error.

## Initial validation sample reviewed

Using the existing `approach1` scratch-CNN checkpoint, I generated and visually inspected an eight-image sample (four per true class) from validation. The CSV and contact sheet are at `outputs/review_readable/review.csv` and `outputs/review_readable/contact_sheet.png` in this local workspace. It contained two correct and two incorrect NORMAL predictions, plus three correct and one incorrect PNEUMONIA prediction. One false-positive NORMAL example had a strong activation near the lower-left image edge, outside the lung field; this is a candidate for human follow-up, not proof that the model uses an artefact. Most other sampled maps were weak or diffuse, so this small sample did not show a consistent, clearly lung-confined focus. Several source images visibly include side markers or other framing/recording marks, but the maps do not establish that the model relies on them. These observations are limited to the listed sample and checkpoint and should not be generalized as performance or clinical findings.

## Review checklist and reporting

Review the contact sheet first, then inspect individual overlays alongside the original X-rays. For each class and for both correct and incorrect predictions when available, note whether high-activation regions appear to overlap lung fields, borders, corners, text markers, image framing, or other conspicuous regions. Record observations, sample IDs, and uncertainty; a highlighted border or marker is a candidate shortcut to investigate, not automatic proof of an artefact. Do not infer patient identity or diagnostic validity from a heatmap.

Grad-CAM is class-score dependent, spatially coarse, and sensitive to target-layer choice and preprocessing. It does not prove which pixels caused a prediction, does not establish that the model understands anatomy, and cannot guarantee patient-level separation. This workflow uses a small balanced validation review sample; findings are qualitative and do not replace larger, blinded review. Do not use official test images to choose target layers, preprocessing, or models. Reserve test evaluation for the final locked model only.

## Tests

```powershell
& $python -m unittest discover -s .\experiments\explainability\tests -v
```

