"""Safely audit exact and possible visual duplicates in existing dataset splits.

The default operation is a report scan. Exact same-label duplicates can be
proposed for quarantine, but moving files requires ``--apply`` and typing
``QUARANTINE``. Perceptual-hash candidates are report-only and never moved.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    from .inspect_data import image_files, sha256_file
    from .data_pipeline import CLASS_TO_LABEL, SPLITS, resolve_dataset_root, download_dataset
except ImportError:  # Supports ``python data/audit_duplicates.py`` from the project root.
    from inspect_data import image_files, sha256_file
    from data_pipeline import CLASS_TO_LABEL, SPLITS, resolve_dataset_root, download_dataset


# A conservative default keeps the first pass reviewable. Broader thresholds
# are available explicitly via --threshold.
DEFAULT_HAMMING_THRESHOLD = 0
DEFAULT_MANIFEST = Path(__file__).with_name("duplicate_audit.csv")
IMAGE_CLASSES = tuple(CLASS_TO_LABEL)
SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for one readable image in the current dataset."""

    path: Path
    split: str
    label: str
    sha256: str
    perceptual_hash: str


@dataclass
class AuditGroup:
    """One exact-duplicate group or perceptual candidate group."""

    group_id: str
    method: str
    records: list
    hash_distances: list
    proposed_keeper: Path | None = None
    proposed_redundant: list | None = None
    status: str = "candidate_report_only"

    @property
    def labels(self):
        return sorted({record.label for record in self.records})

    @property
    def splits(self):
        return sorted({record.split for record in self.records}, key=SPLIT_ORDER.get)

    @property
    def crosses_splits(self):
        return len(self.splits) > 1

    @property
    def crosses_labels(self):
        return len(self.labels) > 1


class BKTree:
    """Small Burkhard-Keller tree for configurable Hamming-distance lookup."""

    def __init__(self):
        self.value = None
        self.indices = []
        self.children = {}

    def add(self, value, index):
        if self.value is None:
            self.value = value
            self.indices.append(index)
            return
        distance = hamming_distance(self.value, value)
        if distance == 0:
            self.indices.append(index)
            return
        child = self.children.setdefault(distance, BKTree())
        child.add(value, index)

    def query(self, value, threshold):
        if self.value is None:
            return []
        distance = hamming_distance(self.value, value)
        matches = list(self.indices) if distance <= threshold else []
        lower = max(1, distance - threshold)
        upper = distance + threshold
        for edge_distance, child in self.children.items():
            if lower <= edge_distance <= upper:
                matches.extend(child.query(value, threshold))
        return matches


def hamming_distance(first, second):
    """Return the number of differing bits in two 64-bit hash strings."""
    if len(first) != len(second):
        raise ValueError("Average hashes must have the same length.")
    return sum(left != right for left, right in zip(first, second))


def dct_perceptual_hash(image_path):
    """Return a 64-bit DCT perceptual hash for one image.

    The image is converted to grayscale and resized to 32x32. A 2D orthonormal
    DCT is computed with NumPy; the low-frequency 8x8 coefficients are reduced
    to bits by comparing each coefficient with the median of those coefficients
    excluding DC. This hash is a similarity screen, not an identity test.
    """
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32)

    size = pixels.shape[0]
    coordinates = np.arange(size, dtype=np.float32)
    basis = np.cos(np.pi / size * (coordinates[:, None] + 0.5) * coordinates[None, :])
    basis[0, :] *= 1 / np.sqrt(2)
    basis *= np.sqrt(2 / size)
    coefficients = basis @ pixels @ basis.T
    low_frequency = coefficients[:8, :8]
    median = np.median(low_frequency.reshape(-1)[1:])
    return "".join("1" if value >= median else "0" for value in low_frequency.reshape(-1))


def scan_images(dataset_root):
    """Scan all expected split/class folders and return records plus read errors."""
    root = Path(dataset_root)
    records = []
    errors = []
    for split in SPLITS:
        for label in IMAGE_CLASSES:
            class_folder = root / split / label
            if not class_folder.is_dir():
                raise FileNotFoundError(f"Missing expected dataset folder: {class_folder}")
            for path in image_files(class_folder):
                try:
                    records.append(ImageRecord(
                        path=path,
                        split=split,
                        label=label,
                        sha256=sha256_file(path),
                        perceptual_hash=dct_perceptual_hash(path),
                    ))
                except (OSError, ValueError) as error:
                    errors.append((path, f"{type(error).__name__}: {error}"))
    records.sort(key=lambda record: (SPLIT_ORDER[record.split], record.label, record.path.as_posix()))
    return records, errors


def exact_groups(records):
    """Group records with identical raw file bytes."""
    by_digest = defaultdict(list)
    for record in records:
        by_digest[record.sha256].append(record)

    groups = []
    for entries in by_digest.values():
        if len(entries) < 2:
            continue
        entries = sorted(entries, key=keeper_sort_key)
        labels = {record.label for record in entries}
        if len(labels) == 1:
            keeper = entries[0]
            redundant = entries[1:]
            status = "proposed_exact_quarantine"
        else:
            keeper = None
            redundant = []
            status = "review_required_different_labels"
        groups.append(AuditGroup(
            group_id="",
            method="exact",
            records=entries,
            hash_distances=[],
            proposed_keeper=keeper.path if keeper else None,
            proposed_redundant=[record.path for record in redundant],
            status=status,
        ))
    return sorted(groups, key=group_sort_key)


def perceptual_groups(records, threshold, exact_groups_found):
    """Group non-exact records whose DCT hashes are within ``threshold`` bits."""
    if threshold < 0 or threshold > 64:
        raise ValueError("Hamming threshold must be between 0 and 64.")

    exact_pairs = {
        frozenset(record.path for record in group.records)
        for group in exact_groups_found
    }
    tree = BKTree()
    parent = list(range(len(records)))
    edges = []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for index, record in enumerate(records):
        for prior_index in tree.query(record.perceptual_hash, threshold):
            prior = records[prior_index]
            pair = frozenset((record.path, prior.path))
            if pair in exact_pairs:
                continue
            distance = hamming_distance(record.perceptual_hash, prior.perceptual_hash)
            union(index, prior_index)
            edges.append((distance, index, prior_index))
        tree.add(record.perceptual_hash, index)

    components = defaultdict(set)
    for distance, first_index, second_index in edges:
        root = find(first_index)
        components[root].update((first_index, second_index))
    component_distances = defaultdict(list)
    for distance, first_index, _ in edges:
        component_distances[find(first_index)].append(distance)
    groups = []
    for indices in components.values():
        if len(indices) < 2:
            continue
        entries = sorted((records[index] for index in indices), key=keeper_sort_key)
        groups.append(AuditGroup(
            group_id="",
            method="perceptual_candidate",
            records=entries,
            hash_distances=sorted(component_distances[find(next(iter(indices)))]),
            status="candidate_report_only",
        ))
    return sorted(groups, key=group_sort_key)


def keeper_sort_key(record):
    """Prefer train, then val, then test, with a stable path tie-breaker."""
    return (SPLIT_ORDER[record.split], record.label, record.path.as_posix())


def group_sort_key(group):
    return keeper_sort_key(sorted(group.records, key=keeper_sort_key)[0])


def audit_dataset(dataset_root, threshold=DEFAULT_HAMMING_THRESHOLD):
    """Scan and return exact groups, perceptual groups, and unreadable files."""
    records, errors = scan_images(dataset_root)
    exact = exact_groups(records)
    candidates = perceptual_groups(records, threshold, exact)
    for index, group in enumerate(exact, start=1):
        group.group_id = f"EXACT-{index:04d}"
    for index, group in enumerate(candidates, start=1):
        group.group_id = f"PERCEPTUAL-{index:04d}"
    return records, exact, candidates, errors


def relative_path(path, dataset_root):
    return Path(path).relative_to(Path(dataset_root)).as_posix()


def group_row(group, dataset_root, threshold):
    distances = ""
    if group.hash_distances:
        distances = ";".join(str(distance) for distance in group.hash_distances)
    return {
        "group_id": group.group_id,
        "detection_method": group.method,
        "hash_algorithm": "sha256_file_bytes" if group.method == "exact" else "dct_phash_32x32_lowfreq_8x8",
        "hamming_threshold": threshold,
        "file_paths": ";".join(relative_path(record.path, dataset_root) for record in group.records),
        "labels": ";".join(group.labels),
        "splits": ";".join(group.splits),
        "hash_distance": distances,
        "crosses_splits": str(group.crosses_splits).lower(),
        "crosses_labels": str(group.crosses_labels).lower(),
        "proposed_keeper": relative_path(group.proposed_keeper, dataset_root) if group.proposed_keeper else "",
        "proposed_redundant": ";".join(
            relative_path(path, dataset_root) for path in (group.proposed_redundant or [])
        ),
        "status": group.status,
    }


def write_manifest(path, dataset_root, records, exact, candidates, errors, threshold):
    """Write the inventory and duplicate-group review CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_id", "detection_method", "hash_algorithm", "hamming_threshold",
        "file_paths", "labels", "splits",
        "hash_distance", "crosses_splits", "crosses_labels", "proposed_keeper",
        "proposed_redundant", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            writer.writerow({
                "group_id": f"IMAGE-{index:06d}",
                "detection_method": "inventory",
                "hash_algorithm": "sha256_file_bytes;dct_phash_32x32_lowfreq_8x8",
                "hamming_threshold": threshold,
                "file_paths": relative_path(record.path, dataset_root),
                "labels": record.label,
                "splits": record.split,
                "status": "scanned",
            })
        for group in [*exact, *candidates]:
            writer.writerow(group_row(group, dataset_root, threshold))
        for image_path, error in errors:
            writer.writerow({
                "group_id": "UNREADABLE",
                "detection_method": "unreadable",
                "hash_algorithm": "",
                "hamming_threshold": threshold,
                "file_paths": relative_path(image_path, dataset_root),
                "status": error,
            })
    return path


def proposed_exact_moves(exact_groups_found):
    """Return only same-label exact groups with safe deterministic proposals."""
    return [
        group for group in exact_groups_found
        if group.status == "proposed_exact_quarantine" and group.proposed_redundant
    ]


def apply_exact_quarantine(groups, dataset_root, quarantine_dir):
    """Move proposed exact redundant files and write a reversible manifest."""
    root = Path(dataset_root)
    quarantine = Path(quarantine_dir)
    moves = []
    for group in groups:
        for original in group.proposed_redundant or []:
            original = Path(original)
            if not original.is_file():
                raise FileNotFoundError(f"Proposed file no longer exists: {original}")
            destination = quarantine / original.relative_to(root)
            if destination.exists():
                raise FileExistsError(f"Quarantine destination already exists: {destination}")
            moves.append((group.group_id, original, destination))

    quarantine.mkdir(parents=True, exist_ok=True)
    manifest_path = quarantine / "quarantine_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["group_id", "original_path", "quarantine_path", "sha256", "moved_at_utc"],
        )
        writer.writeheader()
        for group_id, original, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(original), str(destination))
            writer.writerow({
                "group_id": group_id,
                "original_path": str(original),
                "quarantine_path": str(destination),
                "sha256": sha256_file(destination),
                "moved_at_utc": datetime.now(timezone.utc).isoformat(),
            })
    return manifest_path, len(moves)


def restore_quarantine(manifest_path):
    """Restore files recorded in a quarantine manifest when paths are available."""
    manifest_path = Path(manifest_path)
    restored = 0
    skipped = []
    with manifest_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            original = Path(row["original_path"])
            quarantine = Path(row["quarantine_path"])
            if original.exists():
                skipped.append(f"original already exists: {original}")
                continue
            if not quarantine.is_file():
                skipped.append(f"quarantine file missing: {quarantine}")
                continue
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(quarantine), str(original))
            restored += 1
    return restored, skipped


def print_summary(dataset_root, records, exact, candidates, errors, threshold, manifest):
    print(f"Scanned readable images: {len(records)}")
    print(f"Unreadable images: {len(errors)}")
    print(f"Exact duplicate groups: {len(exact)}")
    print(f"Perceptual candidate groups (Hamming <= {threshold}): {len(candidates)}")
    print(f"Exact groups crossing splits: {sum(group.crosses_splits for group in exact)}")
    print(f"Exact groups crossing labels: {sum(group.crosses_labels for group in exact)}")
    print(f"Perceptual candidates crossing splits: {sum(group.crosses_splits for group in candidates)}")
    print(f"Review manifest: {manifest}")
    proposed = sum(len(group.proposed_redundant or []) for group in exact)
    print(f"Exact same-label files proposed for quarantine: {proposed}")


def main():
    parser = argparse.ArgumentParser(description="Audit exact and possible visual duplicates safely.")
    parser.add_argument("--dataset-dir", type=Path, help="existing dataset directory; otherwise use KaggleHub")
    parser.add_argument("--threshold", type=int, default=DEFAULT_HAMMING_THRESHOLD,
                        help=f"DCT pHash Hamming threshold, default {DEFAULT_HAMMING_THRESHOLD}")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="CSV review manifest path")
    parser.add_argument("--apply", action="store_true",
                        help="after confirmation, quarantine proposed exact same-label redundancies")
    parser.add_argument("--quarantine-dir", type=Path, default=None,
                        help="quarantine destination; defaults beside the dataset root")
    parser.add_argument("--restore-manifest", type=Path,
                        help="restore files listed in a quarantine manifest")
    args = parser.parse_args()

    if args.restore_manifest:
        restored, skipped = restore_quarantine(args.restore_manifest)
        print(f"Restored files: {restored}")
        for message in skipped:
            print(f"Skipped: {message}")
        return 0

    try:
        dataset_root = resolve_dataset_root(args.dataset_dir) if args.dataset_dir else download_dataset()
        records, exact, candidates, errors = audit_dataset(dataset_root, args.threshold)
        manifest = write_manifest(args.manifest, dataset_root, records, exact, candidates, errors, args.threshold)
        print_summary(dataset_root, records, exact, candidates, errors, args.threshold, manifest)

        if not args.apply:
            print("Report only: no files were moved. pHash matches are candidates, not confirmed duplicates.")
            return 0

        proposals = proposed_exact_moves(exact)
        if not proposals:
            print("No safe same-label exact-duplicate quarantine proposals were found.")
            return 0
        total = sum(len(group.proposed_redundant or []) for group in proposals)
        print(f"About to move {total} exact same-label redundant file(s) to quarantine.")
        print("Perceptual candidates and cross-label exact groups will not be moved.")
        confirmation = input("Type QUARANTINE to continue: ")
        if confirmation.strip() != "QUARANTINE":
            print("Confirmation did not match; no files were moved.")
            return 0
        quarantine_dir = args.quarantine_dir or Path(dataset_root).parent / f"{Path(dataset_root).name}_duplicate_quarantine"
        quarantine_manifest, moved = apply_exact_quarantine(proposals, dataset_root, quarantine_dir)
        print(f"Moved {moved} file(s) to quarantine: {quarantine_manifest}")
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
