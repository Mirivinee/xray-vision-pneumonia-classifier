"""Synthetic tests for safe same-class exact duplicate handling and splits."""

import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split_data import (
    find_exact_duplicates,
    quarantine_exact_duplicates,
    restore_quarantine,
    scan_images,
    split_counts,
    write_audit_report,
)


class DuplicateCleanupTests(unittest.TestCase):
    def make_dataset(self, root):
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                (root / split / label).mkdir(parents=True, exist_ok=True)

    def add(self, root, relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        intensity = sum(content) % 256
        Image.new("RGB", (5, 4), (intensity, intensity, intensity)).save(path, format="PNG")
        return path

    def temporary_directory(self):
        # The system temp directory can be blocked by Windows Application Control.
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3])

    def test_same_class_duplicate_and_lexical_keeper(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            self.add(root, "train/NORMAL/z.jpg", b"same-bytes")
            self.add(root, "val/NORMAL/a.jpg", b"same-bytes")
            groups = find_exact_duplicates(scan_images(root))
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["kept"]["relative_path"], "train/NORMAL/z.jpg")
            # Lexicographic path comparison selects the train/ path before val/.
            self.assertEqual(groups[0]["redundant"][0]["relative_path"], "val/NORMAL/a.jpg")

    def test_identical_bytes_in_different_classes_are_not_grouped(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            self.add(root, "train/NORMAL/normal.jpg", b"pixels")
            self.add(root, "train/PNEUMONIA/pneumonia.jpg", b"pixels")
            self.assertEqual(find_exact_duplicates(scan_images(root)), [])

    def test_dry_run_report_does_not_move_any_files(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            first = self.add(root, "train/NORMAL/a.jpg", b"duplicate")
            second = self.add(root, "test/NORMAL/b.jpg", b"duplicate")
            groups = find_exact_duplicates(scan_images(root))
            report_dir = root / "reports"
            write_audit_report(report_dir, scan_images(root), [
                {"group_id": "exact-0001", "method": "SHA256", "label": groups[0]["label"],
                 "sha256": groups[0]["sha256"], "kept_path": groups[0]["kept"]["relative_path"],
                 "quarantined_paths": [item["relative_path"] for item in groups[0]["redundant"]],
                 "copy_count": 2, "redundant_count": 1, "action": "proposed; dry-run"}
            ])
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertFalse((root / ".duplicate_quarantine").exists())
            self.assertTrue((report_dir / "duplicate_cleanup.csv").is_file())

    def test_apply_quarantines_and_rerun_skips_quarantined_copy(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            self.add(root, "train/NORMAL/1.jpg", b"duplicate")
            redundant = self.add(root, "val/NORMAL/2.jpg", b"duplicate")
            groups = find_exact_duplicates(scan_images(root))
            moved = quarantine_exact_duplicates(root, groups, lambda _prompt: True)
            self.assertEqual(moved, 1)
            self.assertFalse(redundant.exists())
            manifest = root / ".duplicate_quarantine" / "manifest.json"
            self.assertTrue(manifest.is_file())
            self.assertEqual(find_exact_duplicates(scan_images(root)), [])
            self.assertEqual(quarantine_exact_duplicates(root, find_exact_duplicates(scan_images(root)),
                                                         lambda _prompt: True), 0)

    def test_restore_uses_manifest_and_split_counts_keep_stratified_ratios(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            self.add(root, "train/NORMAL/a.jpg", b"duplicate")
            redundant = self.add(root, "val/NORMAL/b.jpg", b"duplicate")
            groups = find_exact_duplicates(scan_images(root))
            quarantine_exact_duplicates(root, groups, lambda _prompt: True)
            self.assertEqual(restore_quarantine(root, lambda _prompt: True), 1)
            self.assertTrue(redundant.is_file())
            self.assertEqual(split_counts(1579), {"train": 1263, "val": 158, "test": 158})
            self.assertEqual(split_counts(4245), {"train": 3396, "val": 425, "test": 424})


if __name__ == "__main__":
    unittest.main()

