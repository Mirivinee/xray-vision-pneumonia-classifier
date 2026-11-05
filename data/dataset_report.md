# X-Ray Vision dataset and split report

This report separates the latest local Kaggle folder counts from the project's current generated manifest counts. They describe the same 5,824-image collection organized in two different ways. An older committed report recorded a different 5,856-image dataset snapshot and 30 exact duplicate groups; those historical findings are not treated as findings for the current 5,824-image copy.

## Original dataset folder counts

The inspection recorded these counts in the downloaded dataset's original split folders:

| Original folder | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| `train` | 1,340 | 3,850 | 5,190 |
| `val` | 8 | 8 | 16 |
| `test` | 231 | 387 | 618 |
| **All folders** | **1,579** | **4,245** | **5,824** |

The old inspection also recorded 5,824 readable JPEG files, 4,803 distinct dimensions, and 0 unreadable images. These are inspection findings for the original folders; the counts below are the current project manifest assignments.

## Current shared split manifest

`experiments/data_split_and_imbalance/manifests/splits.csv` is the manifest currently used by the training and explainability experiments. Its summary records an 80/10/10 stratified split with seed 42:

| Manifest split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| `train` | 1,263 | 3,396 | 4,659 |
| `val` | 158 | 425 | 583 |
| `test` | 158 | 424 | 582 |
| **All manifest rows** | **1,579** | **4,245** | **5,824** |

The two tables have the same overall class totals. The manifest reassigns paths among train, validation, and test; it does not describe the source folders' original counts. Patient-level separation is not guaranteed.

## Duplicate and data-quality findings

- The latest same-class SHA-256 dry run scanned 5,824 images and found 0 exact duplicate groups and 0 proposed redundant copies. No images were quarantined.
- The earlier inspection reported 0 exact duplicate groups and 0 unreadable/corrupt images.
- The earlier 8×8 average-hash screen reported 263 candidate groups, including 66 spanning original splits. These are candidates only, not confirmed duplicates or leakage; they were not removed.
- The older committed 5,856-image report listed 30 within-split exact-duplicate groups and 292 average-hash candidate groups. Keep those findings attached to that earlier snapshot; the current local copy was separately scanned and reported 0 exact groups and 263 average-hash candidates.
- Exact-byte checks cannot establish patient-level separation or catch visually identical images encoded differently.

## Pipeline contract

- Labels: `NORMAL = 0`, `PNEUMONIA = 1`.
- Existing model loader preprocessing: RGB conversion, deterministic bilinear resize to 224×224, and float tensor scaling to `[0,1]`; no channel normalization or random augmentation by default.
- The current training scripts use the manifest above and keep test rows out of training/validation model selection.
- The planned shared preparation pipeline will report unreadable files, quarantine only same-class exact duplicates reversibly, and write RGB/grayscale outputs from the same manifest IDs and split assignments. Dimension handling will be optional and configurable.

## History and limitations

The original report incorrectly presented source-folder counts as the current generated split counts and contained stale notes claiming no model files existed. This reconciled report distinguishes the original-folder inspection from current manifest assignments. No new dataset inspection or split regeneration was performed to produce this reconciliation. Hash and perceptual similarity findings do not establish patient identity or clinical validity.
