"""Small standard-library tests for split assignment and augmentation policy."""

import sys
import unittest
import csv
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split_data import assign_splits, check_assignments
from training_common import (
    augmentation_enabled,
    compute_class_weights,
    compute_sample_weights,
    make_dataset_class,
    make_split_loader,
    make_train_val_loaders,
    metrics_from_counts,
    require_torch,
    validate_rotation_degrees,
    validate_strategy,
)
from model_integration import build_partner_model, resolve_factory_spec


def synthetic_rows(normal_count=80, pneumonia_count=120):
    rows = []
    for label, count in (("NORMAL", normal_count), ("PNEUMONIA", pneumonia_count)):
        for index in range(count):
            rows.append({"path": f"{label}/{index}.jpg", "label": label, "sha256": f"{label}-{index}"})
    return rows


class SplitTests(unittest.TestCase):
    def test_seed_makes_assignments_reproducible(self):
        first, second = synthetic_rows(), synthetic_rows()
        assign_splits(first, seed=73)
        assign_splits(second, seed=73)
        self.assertEqual([row["new_split"] for row in first], [row["new_split"] for row in second])

    def test_class_proportions_are_close_to_requested_ratios(self):
        rows = synthetic_rows()
        target, counts = assign_splits(rows, seed=42)
        self.assertEqual(target["NORMAL"], {"train": 64, "val": 8, "test": 8})
        self.assertEqual(target["PNEUMONIA"], {"train": 96, "val": 12, "test": 12})
        self.assertEqual(counts, target)

    def test_exact_duplicates_stay_together_including_across_labels(self):
        rows = synthetic_rows(20, 20)
        rows.extend((
            {"path": "NORMAL/duplicate.jpg", "label": "NORMAL", "sha256": "same-bytes"},
            {"path": "PNEUMONIA/duplicate.jpg", "label": "PNEUMONIA", "sha256": "same-bytes"},
        ))
        assign_splits(rows, seed=9)
        self.assertEqual(check_assignments(rows), len({row["sha256"] for row in rows}))
        pair = [row["new_split"] for row in rows if row["sha256"] == "same-bytes"]
        self.assertEqual(pair[0], pair[1])

    def test_validation_and_test_never_enable_augmentation(self):
        self.assertTrue(augmentation_enabled("train", True))
        self.assertFalse(augmentation_enabled("val", True))
        self.assertFalse(augmentation_enabled("test", True))
        self.assertFalse(augmentation_enabled("train", False))

    def test_weights_are_computed_from_rows_passed_as_training_rows(self):
        rows = ([{"label": "NORMAL"}] * 2) + ([{"label": "PNEUMONIA"}] * 6)
        weights = compute_class_weights(rows)
        self.assertEqual(weights["NORMAL"], 2.0)
        self.assertAlmostEqual(weights["PNEUMONIA"], 2 / 3)
        sample_weights = compute_sample_weights(rows)
        self.assertEqual(sample_weights[:2], [0.5, 0.5])
        self.assertEqual(sample_weights[2:], [1 / 6] * 6)

    def test_imbalance_modes_are_mutually_exclusive(self):
        validate_strategy("class-weighted-loss")
        validate_strategy("weighted-sampling")
        with self.assertRaises(ValueError):
            validate_strategy("class-weighted-loss,weighted-sampling")

    def test_transform_builder_augments_train_but_not_val_or_test(self):
        class Compose:
            def __init__(self, operations):
                self.operations = operations

        class Marker:
            def __init__(self, *args, **kwargs):
                pass

        transforms = SimpleNamespace(
            RandomRotation=Marker,
            Resize=Marker,
            ToTensor=Marker,
            Compose=Compose,
        )
        fake_torch = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)))
        dataset_class = make_dataset_class(fake_torch, object, transforms)
        train_set = dataset_class([], "train", augment_train=True)
        val_set = dataset_class([], "val", augment_train=True)
        test_set = dataset_class([], "test", augment_train=True)
        self.assertIsInstance(train_set.transform.operations[0], Marker)
        self.assertEqual(len(val_set.transform.operations), 2)
        self.assertEqual(len(test_set.transform.operations), 2)

    def test_rotation_range_is_bounded(self):
        validate_rotation_degrees(3)
        with self.assertRaises(ValueError):
            validate_rotation_degrees(11)

    def test_missing_partner_model_has_actionable_message(self):
        with patch.dict("os.environ", {"XRAY_MODEL_FACTORY": ""}):
            with self.assertRaisesRegex(RuntimeError, "Partner model is not configured"):
                resolve_factory_spec()

    def test_partner_factory_receives_two_class_interface(self):
        module = ModuleType("fake_partner")
        module.build_model = lambda num_classes: {"num_classes": num_classes}
        with patch.dict(sys.modules, {"fake_partner": module}):
            model = build_partner_model("fake_partner:build_model")
        self.assertEqual(model, {"num_classes": 2})

    def test_binary_metrics_and_confusion_matrix(self):
        result = metrics_from_counts(8, 2, 1, 9)
        self.assertEqual(result["accuracy"], 17 / 20)
        self.assertEqual(result["confusion_matrix"], [[8, 2], [1, 9]])
        self.assertAlmostEqual(result["precision"], 9 / 11)
        self.assertAlmostEqual(result["recall"], 0.9)

    def test_real_loader_batch_contract_when_torchvision_is_available(self):
        try:
            torch, _, _, _, _ = require_torch()
        except RuntimeError as error:
            self.skipTest(str(error))
        from PIL import Image

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as temporary:
            root = Path(temporary)
            rows = []
            for split in ("train", "val", "test"):
                for label, value in (("NORMAL", 40), ("PNEUMONIA", 210)):
                    relative = Path(split) / label / "sample.png"
                    image_path = root / relative
                    image_path.parent.mkdir(parents=True)
                    Image.new("L", (16, 12), color=value).save(image_path)
                    rows.append({"relative_path": relative.as_posix(), "label": label, "split": split})
            manifest = root / "splits.csv"
            with manifest.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=("relative_path", "label", "split"))
                writer.writeheader()
                writer.writerows(rows)

            _, loaders, _ = make_train_val_loaders(manifest, root, batch_size=2)
            images, labels = next(iter(loaders["val"]))
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(images.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(images).all()))
            self.assertGreaterEqual(float(images.min()), 0.0)
            self.assertLessEqual(float(images.max()), 1.0)
            self.assertEqual(set(labels.tolist()), {0, 1})

            _, test_loader, test_rows = make_split_loader(manifest, root, "test", batch_size=2)
            test_images, _ = next(iter(test_loader))
            self.assertEqual(len(test_rows), 2)
            self.assertEqual(tuple(test_images.shape), (2, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
