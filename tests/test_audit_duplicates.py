"""Synthetic tests for the safe duplicate-audit utility."""

import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from data.audit_duplicates import (
    audit_dataset,
    apply_exact_quarantine,
    dct_perceptual_hash,
    hamming_distance,
    restore_quarantine,
)


class DuplicateAuditTests(unittest.TestCase):
    def setUp(self):
        # The managed Windows Python blocks its redirected system temp folder;
        # keep synthetic fixtures in the writable project workspace instead.
        workspace = Path(__file__).resolve().parents[1]
        self.temp_dir = tempfile.TemporaryDirectory(dir=workspace)
        self.root = Path(self.temp_dir.name) / "dataset"
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                (self.root / split / label).mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def base_image(size=32, brightness=100):
        image = Image.new("L", (size, size), brightness)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, size - 8, size - 8), fill=min(brightness + 50, 255))
        return image

    def save(self, image, split, label, filename, **kwargs):
        path = self.root / split / label / filename
        image.save(path, **kwargs)
        return path

    def test_byte_identical_cross_split_group_has_deterministic_keeper(self):
        source = self.save(self.base_image(), "train", "NORMAL", "source.png")
        duplicate = self.root / "val" / "NORMAL" / "duplicate.png"
        duplicate.write_bytes(source.read_bytes())

        _, exact, _, errors = audit_dataset(self.root, threshold=0)

        self.assertFalse(errors)
        group = next(group for group in exact if len(group.records) == 2)
        self.assertEqual(group.status, "proposed_exact_quarantine")
        self.assertEqual(group.proposed_keeper, source)
        self.assertEqual(group.proposed_redundant, [duplicate])
        self.assertTrue(group.crosses_splits)

    def test_same_bytes_with_different_labels_is_never_proposed(self):
        source = self.save(self.base_image(), "train", "NORMAL", "normal.png")
        duplicate = self.root / "test" / "PNEUMONIA" / "wrong_label.png"
        duplicate.write_bytes(source.read_bytes())

        _, exact, _, _ = audit_dataset(self.root, threshold=0)

        group = next(group for group in exact if len(group.records) == 2)
        self.assertTrue(group.crosses_labels)
        self.assertEqual(group.status, "review_required_different_labels")
        self.assertIsNone(group.proposed_keeper)
        self.assertEqual(group.proposed_redundant, [])

    def test_resized_recompressed_visual_candidate_uses_hamming_threshold(self):
        first = self.save(self.base_image(), "train", "NORMAL", "first.png")
        similar = self.root / "val" / "NORMAL" / "similar.jpg"
        self.base_image(size=48, brightness=103).save(similar, format="JPEG", quality=85)

        distance = hamming_distance(dct_perceptual_hash(first), dct_perceptual_hash(similar))
        self.assertGreater(distance, 0)
        _, exact, candidates, _ = audit_dataset(self.root, threshold=distance)

        self.assertFalse(any(first in [record.path for record in group.records] for group in exact))
        self.assertTrue(any(first in [record.path for record in group.records] for group in candidates))
        distances = [distance for group in candidates for distance in group.hash_distances]
        self.assertTrue(distances)
        self.assertTrue(all(candidate_distance <= distance for candidate_distance in distances))

    def test_hamming_distance_is_configurable(self):
        self.assertEqual(hamming_distance("0" * 64, "1" * 64), 64)
        self.assertEqual(hamming_distance("0" * 64, "0" * 64), 0)

    def test_quarantine_manifest_restores_synthetic_duplicate(self):
        source = self.save(self.base_image(), "train", "NORMAL", "source.png")
        duplicate = self.root / "val" / "NORMAL" / "duplicate.png"
        duplicate.write_bytes(source.read_bytes())
        _, exact, _, _ = audit_dataset(self.root, threshold=0)
        group = next(group for group in exact if group.status == "proposed_exact_quarantine")

        quarantine = Path(self.temp_dir.name) / "quarantine"
        manifest, moved = apply_exact_quarantine([group], self.root, quarantine)

        self.assertEqual(moved, 1)
        self.assertFalse(duplicate.exists())
        self.assertTrue(manifest.exists())
        restored, skipped = restore_quarantine(manifest)
        self.assertEqual(restored, 1)
        self.assertFalse(skipped)
        self.assertTrue(duplicate.exists())

        with manifest.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["original_path"], str(duplicate))


if __name__ == "__main__":
    unittest.main()
