"""Run one forward/loss/backward smoke batch for all requested training modes."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from model_integration import build_model
from training_common import (
    DEFAULT_MANIFEST,
    STRATEGIES,
    class_pos_weight,
    make_training_loaders,
    run_epoch,
    sample_weights,
)
from torch.utils.data import WeightedRandomSampler


def smoke_one_mode(manifest, dataset_root, batch_size, device, strategy=None, augment=False,
                   rotation_degrees=3):
    loaders, train_rows, _ = make_training_loaders(
        manifest, dataset_root, batch_size, strategy=strategy,
        augment_train=augment, rotation_degrees=rotation_degrees,
    )
    if strategy == "weighted-sampling" and not isinstance(loaders["train"].sampler, WeightedRandomSampler):
        raise AssertionError("Weighted-sampling mode did not attach WeightedRandomSampler")
    if loaders["val"].dataset.augmented:
        raise AssertionError("Validation transforms must never use augmentation")
    if bool(loaders["train"].dataset.augmented) != bool(augment):
        raise AssertionError("Train augmentation setting does not match this approach")

    images, labels = next(iter(loaders["train"]))
    if labels.dtype != torch.int64 or not set(labels.tolist()).issubset({0, 1}):
        raise AssertionError(f"Invalid integer labels from manifest: {labels}")
    device_images, device_labels = images.to(device), labels.float().to(device)
    model = build_model().to(device)
    logits = model(device_images)
    if tuple(logits.shape) != (len(labels),):
        raise AssertionError(f"Expected one logit per image; got {tuple(logits.shape)}")
    pos_weight = torch.tensor([class_pos_weight(train_rows)], device=device) if strategy == "class-weighted-loss" else None
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, device_labels)
    if not torch.isfinite(loss):
        raise AssertionError("Smoke loss is not finite")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Backward pass produced no model gradients")
    optimizer.step()
    validation = run_epoch(model, loaders["val"], torch.nn.BCEWithLogitsLoss(), device, max_batches=1)
    print(f"PASS strategy={strategy or 'none'}, augmentation={augment}, batch={tuple(images.shape)}, "
          f"logits={tuple(logits.shape)}, loss={loss.item():.4f}, val-f1={validation['f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rotation-degrees", type=float, default=3)
    args = parser.parse_args()
    if not 0 <= args.rotation_degrees <= 10:
        parser.error("rotation-degrees must be between 0 and 10")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    try:
        for strategy in STRATEGIES:
            smoke_one_mode(args.manifest, args.dataset_dir, args.batch_size, device, strategy=strategy)
        smoke_one_mode(args.manifest, args.dataset_dir, args.batch_size, device,
                       augment=True, rotation_degrees=args.rotation_degrees)
        print("Smoke checks only; no checkpoint/history was written and no full training was run.")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, AssertionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
