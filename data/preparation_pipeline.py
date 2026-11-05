"""Shared exact-cleanup, split-manifest, and RGB/grayscale preparation pipeline."""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_pipeline import CLASS_TO_LABEL, IMAGE_EXTENSIONS, SPLITS
from data.dimension_handler import resize_image
from data.inspect_data import find_perceptual_hash_candidates


RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_SEED = 42
MANIFEST_DIR = PROJECT_ROOT / "experiments" / "data_split_and_imbalance" / "manifests"
QUARANTINE_NAME = ".duplicate_quarantine"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_root):
    """Check image readability and return valid records plus unreadable records."""
    root = Path(dataset_root).resolve()
    missing = [root / split / name for split in SPLITS for name in CLASS_TO_LABEL
               if not (root / split / name).is_dir()]
    if missing:
        raise FileNotFoundError("Missing expected dataset folder(s): " + ", ".join(map(str, missing)))
    images, invalid = [], []
    for split in SPLITS:
        for label in CLASS_TO_LABEL:
            folder = root / split / label
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = path.relative_to(root).as_posix()
                if relative.startswith(f"{QUARANTINE_NAME}/"):
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    images.append({"path": path, "relative_path": relative, "label": label,
                                   "existing_split": split})
                except Exception as error:
                    invalid.append({"relative_path": relative, "label": label, "existing_split": split,
                                    "error": f"{type(error).__name__}: {error}"})
    if not images and not invalid:
        raise ValueError(f"No supported images found under {root}")
    return images, invalid


def scan_images(dataset_root):
    """Compatibility helper returning only readable images."""
    return scan_dataset(dataset_root)[0]


def find_exact_duplicates(images):
    """Group byte-identical readable images within their class only."""
    by_digest = defaultdict(list)
    for record in images:
        by_digest[(record["label"], sha256_file(record["path"]))].append(record)
    groups = []
    for (label, digest), records in by_digest.items():
        if len(records) > 1:
            records.sort(key=lambda item: item["relative_path"])
            groups.append({"label": label, "sha256": digest, "kept": records[0],
                           "redundant": records[1:]})
    return sorted(groups, key=lambda item: (item["label"], item["kept"]["relative_path"]))


def split_counts(length, ratios=RATIOS):
    exact = {name: length * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = length - sum(counts.values())
    order = sorted(ratios, key=lambda name: (-(exact[name] - counts[name]), list(ratios).index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def make_split_rows(images, seed=DEFAULT_SEED):
    """Create stable, class-stratified rows using the project's 80/10/10 policy."""
    rows = []
    for label in CLASS_TO_LABEL:
        class_images = sorted((row for row in images if row["label"] == label),
                              key=lambda row: row["relative_path"])
        if not class_images:
            raise ValueError(f"No readable images found for class {label}")
        random.Random(seed + CLASS_TO_LABEL[label]).shuffle(class_images)
        allocation = split_counts(len(class_images))
        offset = 0
        for split in SPLITS:
            subset = class_images[offset:offset + allocation[split]]
            offset += allocation[split]
            rows.extend({"relative_path": row["relative_path"], "label": label, "split": split}
                        for row in subset)
    return rows


def split_summary(rows):
    counts = Counter((row["label"], row["split"]) for row in rows)
    return {label: {split: counts[(label, split)] for split in SPLITS} for label in CLASS_TO_LABEL}


def write_manifests(dataset_root, images, manifest_dir=MANIFEST_DIR, seed=DEFAULT_SEED, overwrite=False):
    destination = Path(manifest_dir)
    csv_path, summary_path = destination / "splits.csv", destination / "summary.json"
    existing = [path for path in (csv_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Manifest(s) already exist: " + ", ".join(map(str, existing)))
    rows = make_split_rows(images, seed)
    with_counts = {label: {split: 0 for split in SPLITS} for label in CLASS_TO_LABEL}
    for row in rows:
        with_counts[row["label"]][row["split"]] += 1
    destination.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("relative_path", "label", "split"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset_root_at_creation": str(Path(dataset_root).resolve()), "seed": seed,
               "ratios": RATIOS, "patient_level_separation_guaranteed": False,
               "counts": with_counts, "target_counts": with_counts}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, with_counts


def create_manifests(dataset_root, images, manifest_dir=MANIFEST_DIR, seed=DEFAULT_SEED, overwrite=False):
    """Compatibility API for the previous split generator."""
    destination = Path(manifest_dir)
    csv_path, summary_path = destination / "splits.csv", destination / "summary.json"
    rows, counts = write_manifests(dataset_root, images, destination, seed, overwrite)
    return csv_path, summary_path, counts


def quarantine_exact_duplicates(dataset_root, groups, confirm):
    """Reversibly move redundant copies and maintain a restore manifest."""
    root = Path(dataset_root).resolve()
    quarantine = root / QUARANTINE_NAME / "files"
    manifest_path = root / QUARANTINE_NAME / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"moves": []}
    moves = []
    for group in groups:
        for record in group["redundant"]:
            target = quarantine / record["relative_path"]
            if target.exists():
                raise FileExistsError(f"Quarantine destination already exists: {target}")
            moves.append({"original_path": record["relative_path"],
                          "quarantine_path": target.relative_to(root).as_posix(),
                          "label": group["label"], "sha256": group["sha256"], "status": "planned"})
    if not moves:
        return 0
    if not confirm(len(moves)):
        return None
    manifest["moves"].extend(moves)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    moved = []
    try:
        for item in moves:
            source, target = root / item["original_path"], root / item["quarantine_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not source.is_file() or sha256_file(source) != item["sha256"]:
                raise RuntimeError(f"Source changed or disappeared: {source}")
            source.rename(target)
            item["status"] = "quarantined"
            moved.append(item)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        for item in reversed(moved):
            source, target = root / item["original_path"], root / item["quarantine_path"]
            source.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not source.exists():
                target.rename(source)
            item["status"] = "rolled_back"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    return len(moved)


def restore_quarantine(dataset_root, confirm):
    root = Path(dataset_root).resolve()
    manifest_path = root / QUARANTINE_NAME / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No quarantine manifest found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending = [item for item in manifest["moves"]
               if item["status"] in ("planned", "quarantined")
               and (root / item["quarantine_path"]).is_file()]
    conflicts = [item["original_path"] for item in pending if (root / item["original_path"]).exists()]
    if conflicts:
        raise FileExistsError("Restore would overwrite existing path(s): " + ", ".join(conflicts))
    if not pending or not confirm(len(pending)):
        return 0
    for item in pending:
        source, target = root / item["quarantine_path"], root / item["original_path"]
        if sha256_file(source) != item["sha256"]:
            raise RuntimeError(f"Quarantined file hash changed: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        item["status"] = "restored"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(pending)


def variant_id(relative_path):
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]


def write_variants(dataset_root, rows, output_dir, image_size=(224, 224)):
    """Write RGB and grayscale PNG variants derived from exactly the same split rows."""
    root, output = Path(dataset_root).resolve(), Path(output_dir)
    if (output / "rgb").exists() or (output / "grayscale").exists() or (output / "variants.csv").exists():
        raise FileExistsError(f"RGB/grayscale outputs already exist under: {output}")
    records, errors = [], []
    for row in rows:
        source = root / row["relative_path"]
        identifier = variant_id(row["relative_path"])
        rgb_path = output / "rgb" / row["split"] / row["label"] / f"{identifier}.png"
        gray_path = output / "grayscale" / row["split"] / row["label"] / f"{identifier}.png"
        try:
            with Image.open(source) as original:
                rgb = resize_image(original.convert("RGB"), image_size)
                gray = resize_image(original.convert("L"), image_size)
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                gray_path.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(rgb_path, format="PNG")
                gray.save(gray_path, format="PNG")
            records.append({"image_id": identifier, "relative_path": row["relative_path"],
                            "label": row["label"], "split": row["split"],
                            "rgb_path": rgb_path.as_posix(), "grayscale_path": gray_path.as_posix()})
        except Exception as error:
            errors.append({"relative_path": row["relative_path"],
                           "error": f"{type(error).__name__}: {error}"})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "variants.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("image_id", "relative_path", "label", "split",
                                                         "rgb_path", "grayscale_path"))
        writer.writeheader()
        writer.writerows(records)
    return records, errors


def audit_report(dataset_root, images, invalid, groups, near_candidates, processing_errors=()):
    root = Path(dataset_root).resolve()
    class_counts = Counter(row["label"] for row in images)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": "configured with --dataset-dir (absolute local path omitted)",
        "counts": {"readable_images": len(images), "invalid_images": len(invalid),
                   "exact_duplicate_groups": len(groups),
                   "proposed_redundant_files": sum(len(group["redundant"]) for group in groups),
                   "processing_errors": len(processing_errors), "by_class": dict(class_counts)},
        "split_counts": None,
        "invalid_images": invalid,
        "exact_duplicates": [{"class": group["label"], "sha256": group["sha256"],
                               "kept_path": group["kept"]["relative_path"],
                               "quarantined_paths": [item["relative_path"] for item in group["redundant"]]}
                              for group in groups],
        "near_duplicate_candidates": [
            {"hash": candidate["hash"], "cross_split": candidate["cross_split"],
             "paths": [{"path": path.relative_to(root).as_posix(), "split": split, "label": label}
                       for path, split, label in candidate["entries"]]}
            for candidate in near_candidates
        ],
        "near_duplicate_method": "8x8 average hash; candidates only, no automatic action",
        "processing_errors": list(processing_errors),
    }


def report_records(groups, applied=False):
    """Compatibility report rows for the former split_data module API."""
    return [{"group_id": f"exact-{index:04d}", "method": "SHA-256 raw file bytes; same class only",
             "label": group["label"], "sha256": group["sha256"],
             "kept_path": group["kept"]["relative_path"],
             "quarantined_paths": [item["relative_path"] for item in group["redundant"]],
             "copy_count": len(group["redundant"]) + 1,
             "redundant_count": len(group["redundant"]),
             "action": "quarantined" if applied else "proposed; dry-run"}
            for index, group in enumerate(groups, 1)]


def write_audit_report(report_dir, images, records):
    """Compatibility CSV/JSON report writer for existing scripts and tests."""
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counts = {"scanned_images": len(images), "exact_duplicate_groups": len(records),
              "proposed_or_quarantined_files": sum(row["redundant_count"] for row in records)}
    payload = {"counts": counts, "exact_duplicate_groups": records,
               "near_duplicate_candidates": "Report-only; see data/inspect_data.py"}
    (destination / "duplicate_cleanup.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ("group_id", "method", "label", "sha256", "kept_path", "quarantined_paths",
              "copy_count", "redundant_count", "action")
    with (destination / "duplicate_cleanup.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "quarantined_paths": json.dumps(row["quarantined_paths"])})
    return counts


def write_report(report, output_dir, overwrite=False):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    text_path, json_path, csv_path = (destination / "data_report.txt",
                                      destination / "preparation_report.json",
                                      destination / "exact_duplicates.csv")
    existing = [path for path in (text_path, json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Report files exist; pass --overwrite-report to refresh: " + ", ".join(map(str, existing)))
    lines = ["X-Ray Vision data preparation report", "", f"Dataset root: {report['dataset_root']}",
             f"Readable images: {report['counts']['readable_images']}",
             f"Invalid/unreadable images: {report['counts']['invalid_images']}",
             f"Readable images remaining after proposed/actual cleanup: {report['counts'].get('remaining_images_after_cleanup', 'not computed')}",
             f"Exact duplicate groups: {report['counts']['exact_duplicate_groups']}",
             f"Redundant exact copies proposed/quarantined: {report['counts']['proposed_redundant_files']}",
             f"Near-duplicate candidates (average hash; review only): {len(report['near_duplicate_candidates'])}",
             f"Processing errors: {report['counts']['processing_errors']}", "",
             "Class counts (readable): " + json.dumps(report['counts']['by_class'], sort_keys=True),
             "Manifest split counts: " + json.dumps(report.get("split_counts"), sort_keys=True),
             "", "Exact duplicate groups:"]
    lines.append(f"Dimension handling: {report.get('dimension_handling', 'configured at run time')}")
    for group in report["exact_duplicates"]:
        lines.append(f"- {group['class']} SHA-256 {group['sha256']}: keep {group['kept_path']}; "
                     f"redundant {', '.join(group['quarantined_paths'])}")
    if not report["exact_duplicates"]:
        lines.append("- None found.")
    if report["invalid_images"]:
        lines.extend(["", "Unreadable images (reported and left unchanged):"])
        lines.extend(f"- {item['relative_path']}: {item['error']}" for item in report["invalid_images"])
    if report["processing_errors"]:
        lines.extend(["", "Variant processing errors:"])
        lines.extend(f"- {item['relative_path']}: {item['error']}" for item in report["processing_errors"])
    lines.extend(["", "Near-duplicate matches are candidates only and are never quarantined automatically.",
                  "Patient-level separation is not guaranteed."])
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ("class", "sha256", "kept_path", "quarantined_paths", "redundant_count")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for group in report["exact_duplicates"]:
            writer.writerow({**group, "quarantined_paths": json.dumps(group["quarantined_paths"]),
                             "redundant_count": len(group["quarantined_paths"])})
    return text_path, json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="optional separate folder for data_report.txt and audit files")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(224, 224))
    parser.add_argument("--no-resize", action="store_true", help="preserve source image dimensions")
    parser.add_argument("--apply", action="store_true", help="quarantine duplicates and create manifests/variants")
    parser.add_argument("--overwrite-manifests", action="store_true")
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--restore", action="store_true", help="restore previously quarantined files")
    args = parser.parse_args()
    try:
        if args.restore:
            restored = restore_quarantine(args.dataset_dir, lambda count: input(
                f"Restore {count} quarantined file(s)? Type RESTORE: "
            ) == "RESTORE")
            print(f"Restored {restored} file(s).")
            return
        root = args.dataset_dir.resolve()
        images, invalid = scan_dataset(root)
        groups = find_exact_duplicates(images)
        near = find_perceptual_hash_candidates([
            (record["path"], record["existing_split"], record["label"]) for record in images
        ])
        report_dir = args.report_dir or args.output_dir
        report = audit_report(root, images, invalid, groups, near)
        report["dimension_handling"] = "disabled" if args.no_resize else f"bilinear resize to {tuple(args.image_size)}"
        redundant = {item["relative_path"] for group in groups for item in group["redundant"]}
        planned_images = [item for item in images if item["relative_path"] not in redundant]
        planned_rows = make_split_rows(planned_images, args.seed)
        report["counts"]["remaining_images_after_cleanup"] = len(planned_images)
        report["split_counts"] = split_summary(planned_rows)
        write_report(report, report_dir, overwrite=args.overwrite_report)
        print(json.dumps(report["counts"], indent=2))
        print(f"Report: {report_dir}")
        if not args.apply:
            print("Dry run only; images, manifests, and variant outputs were not changed.")
            return
        manifest_paths = (args.manifest_dir / "splits.csv", args.manifest_dir / "summary.json")
        if any(path.exists() for path in manifest_paths) and not args.overwrite_manifests:
            raise FileExistsError("Manifests exist; pass --overwrite-manifests and confirm to refresh them")
        if args.overwrite_manifests and input("Overwrite existing manifests? Type OVERWRITE: ") != "OVERWRITE":
            print("Cancelled before modifying images or manifests.")
            return
        output_dir = args.output_dir
        if any((output_dir / name).exists() for name in ("rgb", "grayscale", "variants.csv")):
            raise FileExistsError(f"Variant outputs already exist under: {output_dir}")
        moved = quarantine_exact_duplicates(root, groups, lambda count: input(
            f"Move {count} redundant same-class exact duplicate(s) to quarantine? Type QUARANTINE: "
        ) == "QUARANTINE")
        if moved is None:
            print("Cancelled before writing manifests or variants.")
            return
        images, invalid = scan_dataset(root)
        rows = make_split_rows(images, args.seed)
        if any(path.exists() for path in manifest_paths) and not args.overwrite_manifests:
            raise FileExistsError("Manifests appeared during processing; refusing to overwrite")
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
        with manifest_paths[0].open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=("relative_path", "label", "split"))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"dataset_root_at_creation": str(root), "seed": args.seed, "ratios": RATIOS,
                   "patient_level_separation_guaranteed": False,
                   "counts": split_summary(rows), "target_counts": split_summary(rows)}
        manifest_paths[1].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        variant_rows, processing_errors = write_variants(
            root, rows, output_dir, None if args.no_resize else args.image_size
        )
        report = audit_report(root, images, invalid, groups, near, processing_errors)
        report["counts"]["quarantined_files"] = moved
        report["counts"]["remaining_images_after_cleanup"] = len(images)
        report["counts"]["generated_variant_pairs"] = len(variant_rows)
        report["split_counts"] = split_summary(rows)
        report["dimension_handling"] = "disabled" if args.no_resize else f"bilinear resize to {tuple(args.image_size)}"
        write_report(report, output_dir, overwrite=True)
        print(f"Quarantined {moved}; wrote manifests, {len(variant_rows)} RGB/grayscale pairs, and report to {output_dir}")
        if processing_errors:
            print(f"Image processing errors: {len(processing_errors)} (see data_report.txt)")
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

