# Baseline Scratch CNN: Results and Known Problems

Owner: Suraj. Project: X-Ray Vision (pediatric chest X-ray, NORMAL vs PNEUMONIA).
Academic project, not a medical diagnostic tool.

## 1. What was trained

| Item | Value |
|---|---|
| Model | `ScratchCNN` in `scratch_cnn.py`: 4 x (Conv3x3, BatchNorm, ReLU, MaxPool2), global average pooling, dropout, 1 linear unit |
| Channels | 32, 64, 128, 256 |
| Parameters | 389,153 |
| Output | 1 logit per image; sigmoid gives P(PNEUMONIA); labels NORMAL=0, PNEUMONIA=1 |
| Input | `[B, 3, 224, 224]` float32 in [0, 1] from `data_pipeline.py`; no ImageNet normalization, no augmentation |
| Loss | `BCEWithLogitsLoss`, no class weighting |
| Optimizer | Adam, lr 1e-3, weight decay 1e-4 |
| Dropout / batch size / epochs / seed | 0.5 / 32 / 15 / 42 |
| Data | Existing splits unchanged: train 5216 (1341 NORMAL, 3875 PNEUMONIA), val 16 (8/8) |
| Test set | Not used |
| Checkpoint kept | `best.pt` = lowest validation loss (epoch 9) |
| Decision threshold | 0.5 for all reported metrics |

Reproduce (from the project root):

```
python cnn_scratch/train_scratch_cnn.py --epochs 15
python cnn_scratch/plot_training_curves.py
```

## 2. Results

Training metrics are running averages over each epoch (dropout and BatchNorm in training mode), not a separate evaluation pass.

| Epoch | Train loss | Train acc | Train recall | Train spec. | Train AUC |
|---|---|---|---|---|---|
| 1 | 0.299 | 0.862 | 0.923 | 0.687 | 0.931 |
| 5 | 0.129 | 0.953 | 0.971 | 0.901 | 0.987 |
| 9 | 0.094 | 0.964 | 0.978 | 0.924 | 0.993 |
| 15 | 0.076 | 0.971 | 0.982 | 0.940 | 0.995 |

Validation (16 images, threshold 0.5):

| Epoch | Val loss | Val acc | Recall | Specificity | AUC |
|---|---|---|---|---|---|
| 1 | 2.293 | 0.500 | 1.000 | 0.000 | 0.734 |
| 5 | 0.236 | 0.938 | 0.875 | 1.000 | 0.953 |
| 9 (best val loss, saved) | 0.217 | 0.938 | 0.875 | 1.000 | 0.984 |
| 12 | 2.767 | 0.500 | 1.000 | 0.000 | 0.875 |
| 15 | 1.051 | 0.563 | 1.000 | 0.125 | 1.000 |

Figure: `training_curves.png`. Raw per-epoch numbers: `history.json`.

Summary:
- The model trains end-to-end and learns the training data well (about 97% train accuracy, 0.995 train AUC).
- Validation accuracy swings between 0.50 and 0.94 across epochs, so there is no reliable estimate of generalization yet.
- No test-set result exists yet. The test set stays untouched until the final evaluation.

## 3. Known problems

1. **Validation set is too small (16 images).** One image changes accuracy by 6.25 points. Epoch-to-epoch comparison, early stopping and checkpoint choice are all unreliable. This is the main blocker.
2. **Unstable validation behaviour.** In many epochs the model predicts PNEUMONIA for every validation image (recall 1.0, specificity 0.0). Validation AUC is still 0.88-1.0, so the ranking is good and the 0.5 threshold is the likely issue. The threshold should be chosen on a larger validation set.
3. **Class imbalance and prior shift.** Train is 74% PNEUMONIA, val is 50%, test is 62.5%. The model leans toward predicting PNEUMONIA. Class weighting and oversampling have not been tried yet (Day 3).
4. **`best.pt` is a weak selection.** It was chosen by val loss on 16 images, so epoch 9 is partly luck. Do not quote its validation numbers as the model's performance.
5. **Possible data leakage is not ruled out.** The dataset report found 30 exact-duplicate groups (all within a single split) and 66 perceptual-hash candidate groups spanning splits (screening only, not confirmed). Patient-level separation between train, val and test has not been verified.
6. **No augmentation yet**, so overfitting is likely as training continues (train loss keeps falling while validation is erratic).
7. **Training stability not tuned.** Only one learning rate (1e-3) and no scheduler were tried. BatchNorm statistics may also add noise to validation.
8. **Training metrics are running averages**, not a clean evaluation pass over the training set.
9. **Single run, single seed.** No variance estimate.

## 4. Planned next steps

- Agree a larger, patient-grouped validation split with Vineet and Anurag (keep the official test set untouched).
- Day 3: compare class weighting (`--class-weight`) and oversampling, then add augmentation, and compare each against this baseline.
- Try a lower learning rate (for example 3e-4) or a scheduler.
- Select the decision threshold on validation data only, then run the final test evaluation once (Anurag, Day 4).

## 5. AI usage disclosure

The model, training script and plotting script were drafted with AI assistance (Claude), then run and reviewed by Suraj.
