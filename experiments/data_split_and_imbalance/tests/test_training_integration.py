"""Focused tests for the manifest loader and ScratchCNN binary interface."""

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from model_integration import build_model
from training_common import (
    LABEL_TO_ID,
    class_counts,
    class_pos_weight,
    compute_training_pixel_statistics,
    make_test_loader,
    make_training_loaders,
    sample_weights,
)
from torch.utils.data import WeightedRandomSampler


class TrainingIntegrationTests(unittest.TestCase):
    def make_fixture(self, root):
        rows = []
        for split, count_per_class in (("train", 3), ("val", 2), ("test", 2)):
            for label, intensity in (("NORMAL", 35), ("PNEUMONIA", 210)):
                for index in range(count_per_class):
                    relative = Path(split) / label / f"sample-{index}.png"
                    image_path = root / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (18, 12), intensity + index).save(image_path)
                    rows.append({"relative_path": relative.as_posix(), "label": label, "split": split})
        manifest = root / "splits.csv"
        with manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=("relative_path", "label", "split"))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_teammate_model_has_one_binary_logit_and_bce_backward(self):
        model = build_model(dropout=0.0)
        images = torch.rand(2, 3, 224, 224)
        labels = torch.tensor([LABEL_TO_ID["NORMAL"], LABEL_TO_ID["PNEUMONIA"]], dtype=torch.float32)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_weights_and_stats_use_training_partition(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as temporary:
            root = Path(temporary)
            manifest = self.make_fixture(root)
            loaders, train_rows, _ = make_training_loaders(manifest, root, batch_size=2)
            self.assertEqual(class_counts(train_rows), {"NORMAL": 3, "PNEUMONIA": 3})
            self.assertEqual(class_pos_weight(train_rows), 1.0)
            self.assertEqual(sample_weights(train_rows), [1 / 3] * 6)
            stats = compute_training_pixel_statistics(train_rows, batch_size=2)
            self.assertEqual(stats["pixel_count"], 6 * 224 * 224)
            self.assertEqual(tuple(next(iter(loaders["val"]))[0].shape), (2, 3, 224, 224))

    def test_sampler_and_augmentation_are_training_only(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as temporary:
            root = Path(temporary)
            manifest = self.make_fixture(root)
            plain, _, _ = make_training_loaders(manifest, root, batch_size=2, strategy="class-weighted-loss")
            sampled, _, _ = make_training_loaders(manifest, root, batch_size=2, strategy="weighted-sampling")
            augmented, _, _ = make_training_loaders(
                manifest, root, batch_size=2, augment_train=True, rotation_degrees=3
            )
            self.assertNotIsInstance(plain["train"].sampler, WeightedRandomSampler)
            self.assertIsInstance(sampled["train"].sampler, WeightedRandomSampler)
            self.assertFalse(plain["val"].dataset.augmented)
            self.assertTrue(augmented["train"].dataset.augmented)
            self.assertFalse(augmented["val"].dataset.augmented)

            first = augmented["val"].dataset[0][0]
            second = augmented["val"].dataset[0][0]
            self.assertTrue(torch.equal(first, second))
            test_loader, _ = make_test_loader(manifest, root, batch_size=2)
            self.assertFalse(test_loader.dataset.augmented)


if __name__ == "__main__":
    unittest.main()
