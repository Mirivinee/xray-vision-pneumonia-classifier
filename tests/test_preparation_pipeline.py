"""Focused synthetic coverage for shared preparation and image variants."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.dimension_handler import resize_image
from data.preparation_pipeline import (
    find_exact_duplicates,
    make_split_rows,
    scan_dataset,
    variant_id,
    write_variants,
)


class PreparationPipelineTests(unittest.TestCase):
    def tempdir(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])

    def make_layout(self, root):
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                (root / split / label).mkdir(parents=True, exist_ok=True)

    def add_image(self, root, relative, color):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (12, 8), color).save(path, format="PNG")
        return path

    def test_scan_reports_corrupt_image_and_duplicates_stay_in_class(self):
        with self.tempdir() as temporary:
            root = Path(temporary)
            self.make_layout(root)
            self.add_image(root, "train/NORMAL/a.png", 50)
            self.add_image(root, "val/NORMAL/b.png", 50)
            self.add_image(root, "train/PNEUMONIA/c.png", 50)
            (root / "test/PNEUMONIA/bad.png").write_bytes(b"not an image")
            images, invalid = scan_dataset(root)
            groups = find_exact_duplicates(images)
            self.assertEqual(len(invalid), 1)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["label"], "NORMAL")
            self.assertEqual(groups[0]["kept"]["relative_path"], "train/NORMAL/a.png")

    def test_rgb_and_gray_outputs_share_ids_and_splits(self):
        with self.tempdir() as temporary:
            root = Path(temporary)
            self.make_layout(root)
            self.add_image(root, "train/NORMAL/a.png", 25)
            self.add_image(root, "train/PNEUMONIA/b.png", 220)
            images, _ = scan_dataset(root)
            rows = make_split_rows(images, seed=42)
            output = root / "processed"
            records, errors = write_variants(root, rows, output, image_size=(16, 10))
            self.assertFalse(errors)
            self.assertEqual(len(records), len(rows))
            for record in records:
                self.assertEqual(record["image_id"], variant_id(record["relative_path"]))
                self.assertIn(record["split"], ("train", "val", "test"))
                with Image.open(record["rgb_path"]) as rgb, Image.open(record["grayscale_path"]) as gray:
                    self.assertEqual(rgb.mode, "RGB")
                    self.assertEqual(gray.mode, "L")
                    self.assertEqual(rgb.size, (16, 10))
                    self.assertEqual(gray.size, (16, 10))

    def test_dimensions_can_be_disabled_or_configured(self):
        source = Image.new("RGB", (11, 7))
        self.assertEqual(resize_image(source).size, (11, 7))
        self.assertEqual(resize_image(source, (20, 13)).size, (20, 13))
        with self.assertRaises(ValueError):
            resize_image(source, (0, 13))


if __name__ == "__main__":
    unittest.main()
