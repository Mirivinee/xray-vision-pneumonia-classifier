"""Preview and, after confirmation, write reproducible split manifests.

This utility reads images from the existing dataset and never copies or moves
them. The default command is report-only; final manifests require --apply and
interactive confirmation. Exact SHA-256 groups stay in one split. pHash
candidates remain informational and do not block splitting.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_AUDIT = Path(__file__).resolve().parents[2] / "data" / "duplicate_audit.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "manifests"


def resolve_dataset_root(dataset_dir=None):
    """Use a supplied root, or the repository's existing KaggleHub resolver."""
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()
    try:
        repository_root = Path(__file__).resolve().parents[2]
        if str(repository_root) not in sys.path:
            sys.path.insert(0, str(repository_root))
        from data.data_pipeline import download_dataset
    except Exception as error:
        raise RuntimeError(
            "Could not load the project's KaggleHub path helper. Install the "
            "documented dependencies or pass --dataset-dir with the dataset root."
        ) from error
    return Path(download_dataset()).resolve()


def sha256_file(path):
    """Hash file bytes in chunks so duplicate grouping needs no image library."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_root):
    """Return records for the existing split folders and report missing files."""
    root = Path(dataset_root)
    missing_folders = [
        root / split / label
        for split in SPLITS
        for label in CLASSES
        if not (root / split / label).is_dir()
    ]
    if missing_folders:
        paths = "\n".join(f"  - {path}" for path in missing_folders)
        raise FileNotFoundError(f"Missing expected dataset folders:\n{paths}")

    records = []
    missing_files = []
    for split in SPLITS:
        for label in CLASSES:
            folder = root / split / label
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if not path.exists():
                    missing_files.append(path)
                    continue
                records.append({
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "sha256": sha256_file(path),
                })
    if missing_files:
        raise FileNotFoundError(f"Files disappeared during scan: {missing_files}")
    if not records:
        raise ValueError(f"No supported images found under {root}")
    return records


def target_counts(total, ratios=RATIOS):
    """Allocate integer split counts using the largest-remainder method."""
    raw = [total * ratio for ratio in ratios]
    result = [int(value) for value in raw]
    remainder = total - sum(result)
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - result[i], reverse=True)[:remainder]:
        result[index] += 1
    return dict(zip(SPLITS, result))


def assign_splits(records, seed=SEED):
    """Stratify classes while keeping every same-byte group in one split."""
    unknown = sorted({record["label"] for record in records} - set(CLASSES))
    if unknown:
        raise ValueError(f"Unexpected labels: {', '.join(unknown)}")

    groups_by_hash = defaultdict(list)
    for record in records:
        groups_by_hash[record["sha256"]].append(record)

    target = {label: target_counts(sum(row["label"] == label for row in records)) for label in CLASSES}
    current = {label: Counter({split: 0 for split in SPLITS}) for label in CLASSES}
    rng = random.Random(seed)
    units = list(groups_by_hash.values())
    rng.shuffle(units)
    units.sort(key=len, reverse=True)

    # Place each exact-byte group where it causes the smallest normalized class-count error.
    for unit in units:
        counts = Counter(record["label"] for record in unit)
        choices = list(SPLITS)
        rng.shuffle(choices)

        def cost(split):
            total_cost = 0.0
            for label in CLASSES:
                wanted = target[label][split]
                after = current[label][split] + counts[label]
                scale = max(wanted, 1)
                total_cost += ((after - wanted) / scale) ** 2 - ((current[label][split] - wanted) / scale) ** 2
            return total_cost

        selected = min(choices, key=cost)
        for record in unit:
            record["new_split"] = selected
            current[record["label"]][selected] += 1

    return target, current


def check_assignments(records):
    """Verify every input is assigned and exact-byte groups stay together."""
    by_hash = defaultdict(set)
    for record in records:
        if record.get("new_split") not in SPLITS:
            raise AssertionError(f"Image has no valid new split: {record['path']}")
        by_hash[record["sha256"]].add(record["new_split"])
    crossing = [digest for digest, splits in by_hash.items() if len(splits) > 1]
    if crossing:
        raise AssertionError(f"Exact duplicate groups cross proposed splits: {len(crossing)}")
    return len(by_hash)


def audit_summary(audit_path):
    """Read group counts from the existing review CSV without treating candidates as truth."""
    path = Path(audit_path)
    if not path.is_file():
        raise FileNotFoundError(f"Duplicate-audit report not found: {path}")
    counts = Counter()
    candidate_cross_split = 0
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            method = row.get("detection_method", "")
            if method in {"exact", "perceptual_candidate"}:
                counts[method] += 1
                if method == "perceptual_candidate" and row.get("crosses_splits", "").lower() == "true":
                    candidate_cross_split += 1
    return {"path": str(path), "exact_groups": counts["exact"],
            "perceptual_candidate_groups": counts["perceptual_candidate"],
            "cross_split_candidates": candidate_cross_split}


def print_report(root, records, target, counts, audit, seed):
    """Print the proposed split summary; no manifest is written in report-only mode."""
    source_counts = Counter(row["label"] for row in records)
    repeated = defaultdict(list)
    for row in records:
        repeated[row["sha256"]].append(row)
    exact_groups = [rows for rows in repeated.values() if len(rows) > 1]
    exact_cross_class = sum(len({row["label"] for row in group}) > 1 for group in exact_groups)

    print(f"Dataset root: {root}")
    print("Mode: report only; no final manifests or dataset changes")
    print(f"Proposed ratios: train/val/test = {RATIOS[0]:.0%}/{RATIOS[1]:.0%}/{RATIOS[2]:.0%}")
    print(f"Fixed random seed: {seed}")
    print(f"Images scanned: {len(records)}")
    print("\nClass counts (overall => proposed train / val / test):")
    for label in CLASSES:
        proposed = " / ".join(str(counts[label][split]) for split in SPLITS)
        desired = " / ".join(str(target[label][split]) for split in SPLITS)
        print(f"  {label}: {source_counts[label]} => {proposed} (targets: {desired})")
    split_totals = {
        split: sum(counts[label][split] for label in CLASSES)
        for split in SPLITS
    }
    print("Proposed split totals: " + ", ".join(f"{split}={split_totals[split]}" for split in SPLITS))
    if sum(split_totals.values()) != len(records):
        raise AssertionError("Proposed split totals do not equal the scanned image count.")
    print(f"Exact SHA-256 duplicate groups: {len(exact_groups)}; cross-label groups: {exact_cross_class}")
    print("Proposed exact-duplicate groups crossing splits: 0 (checked)")
    print(f"Existing audit: {audit['path']}")
    print(f"  Exact groups in audit: {audit['exact_groups']}")
    print(f"  Perceptual candidate groups (report only, not grouped): {audit['perceptual_candidate_groups']}")
    print(f"  Cross-split perceptual candidates (informational): {audit['cross_split_candidates']}")
    print("Patient-level separation: cannot be guaranteed; patient IDs/group metadata were not found in the repository.")
    print("pHash matches are candidates only; they do not block the split or count as confirmed duplicates.")
    print("To write manifests, rerun with --apply and confirm at the prompt.")


def write_manifests(records, output_dir, root, seed, counts, target):
    """Write one combined relative-path CSV and summary JSON without overwriting."""
    output_dir = Path(output_dir)
    csv_path = output_dir / "splits.csv"
    json_path = output_dir / "summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(
            f"Manifest output already exists in {output_dir}. Choose another --output-dir; existing files are never overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("relative_path", "label", "split"))
        writer.writeheader()
        for row in sorted(records, key=lambda item: (item["new_split"], item["label"], item["path"])):
            writer.writerow({"relative_path": row["path"], "label": row["label"], "split": row["new_split"]})
    summary = {
        "dataset_root_at_creation": str(root),
        "seed": seed,
        "ratios": dict(zip(SPLITS, RATIOS)),
        "patient_level_separation_guaranteed": False,
        "counts": {label: dict(counts[label]) for label in CLASSES},
        "target_counts": target,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview or write stratified split manifests without changing images.")
    parser.add_argument("--dataset-dir", type=Path, help="existing dataset root; otherwise resolve through KaggleHub")
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT, help="duplicate-audit CSV requiring review")
    parser.add_argument("--seed", type=int, default=SEED, help=f"fixed shuffle seed (default: {SEED})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="destination used only with --apply")
    parser.add_argument("--apply", action="store_true", help="write final manifests after duplicate review and confirmation")
    args = parser.parse_args(argv)

    try:
        root = resolve_dataset_root(args.dataset_dir)
        audit = audit_summary(args.audit_report)
        records = scan_dataset(root)
        target, counts = assign_splits(records, args.seed)
        check_assignments(records)
        # Re-run the seeded algorithm to make repeatability part of every preview/apply check.
        repeat_records = [dict(row) for row in records]
        assign_splits(repeat_records, args.seed)
        if [(r["path"], r["new_split"]) for r in records] != [(r["path"], r["new_split"]) for r in repeat_records]:
            raise AssertionError("Split assignment is not reproducible for the selected seed.")
        print_report(root, records, target, counts, audit, args.seed)

        if not args.apply:
            return 0
        if (args.output_dir / "splits.csv").exists() or (args.output_dir / "summary.json").exists():
            raise FileExistsError(f"Manifest output already exists in {args.output_dir}; choose a new --output-dir.")
        confirmation = input("Type CREATE to write split manifests (images will not be moved): ")
        if confirmation.strip() != "CREATE":
            print("Confirmation did not match; no manifests were written.")
            return 0
        csv_path, json_path = write_manifests(records, args.output_dir, root, args.seed, counts, target)
        print(f"Wrote split manifest: {csv_path}")
        print(f"Wrote split summary: {json_path}")
        print("Dataset images remain in their original locations.")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError, AssertionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
