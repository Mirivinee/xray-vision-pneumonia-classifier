"""Inspect existing dataset splits and write a shareable Markdown report."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from data_pipeline import download_dataset, resolve_dataset_root


CLASSES = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SAMPLES_PER_CLASS = 2


def image_files(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(image_path):
    """Return format and dimensions, or an error if Pillow cannot verify it."""
    try:
        with Image.open(image_path) as image:
            image_format = image.format or image_path.suffix.lstrip(".").upper()
            dimensions = image.size
            image.verify()
        return image_format, dimensions, None
    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def sha256_file(image_path):
    """Calculate an exact byte-for-byte checksum in small chunks."""
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_counts_text(counts):
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "none"


def dimension_counts_text(counts, limit=8):
    if not counts:
        return "none"
    common = counts.most_common(limit)
    details = ", ".join(f"{width}x{height}={count}" for (width, height), count in common)
    remaining = len(counts) - len(common)
    if remaining:
        details += f", ... ({remaining} other dimension(s))"
    return f"{len(counts)} distinct size(s): {details}"


def class_balance_text(counts):
    normal = counts.get("NORMAL", 0)
    pneumonia = counts.get("PNEUMONIA", 0)
    total = normal + pneumonia
    if not total:
        return "No images available for a class-balance calculation."
    majority, minority = max(normal, pneumonia), min(normal, pneumonia)
    ratio = f"{majority / minority:.2f}:1" if minority else "undefined (one class has 0 images)"
    return (
        f"NORMAL {normal} ({normal / total:.1%}); PNEUMONIA {pneumonia} "
        f"({pneumonia / total:.1%}); majority/minority ratio {ratio}."
    )


def find_exact_duplicates(paths_by_file):
    """Return byte-identical groups and their split/class locations."""
    by_hash = defaultdict(list)
    for image_path, split, class_name in paths_by_file:
        try:
            by_hash[sha256_file(image_path)].append((image_path, split, class_name))
        except OSError as error:
            # Reads can fail after initial discovery (for example, a file is locked).
            print(f"Could not hash {image_path}: {error}")

    duplicates = []
    for digest, entries in by_hash.items():
        if len(entries) > 1:
            locations = {split for _, split, _ in entries}
            duplicates.append({"hash": digest, "entries": entries, "cross_split": len(locations) > 1})
    duplicates.sort(key=lambda group: group["entries"][0][0].as_posix())
    return duplicates


def show_samples(sample_paths):
    if not sample_paths:
        print("\nNo readable sample images are available to display.")
        return
    print("\nOpening a few sample images in your default image viewer:")
    for class_name, image_path in sample_paths:
        print(f"  {class_name}: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.show(title=f"{class_name} - {image_path.name}")
        except Exception as error:
            print(f"  Could not display this image: {error}")


def inspect_dataset(dataset_root, display_samples=False):
    dataset_root = Path(dataset_root)
    counts = Counter()
    formats = Counter()
    dimensions = Counter()
    unreadable = []
    missing_folders = []
    file_records = []
    samples = []

    print("\n=== Dataset inspection ===")
    print(f"Dataset root: {dataset_root}")

    for split in SPLITS:
        print(f"\n{split.upper()} split")
        split_path = dataset_root / split
        if not split_path.is_dir():
            missing_folders.append(split_path)
            print(f"  MISSING: {split_path}")
            continue

        for class_name in CLASSES:
            class_path = split_path / class_name
            if not class_path.is_dir():
                missing_folders.append(class_path)
                print(f"  {class_name}: MISSING folder")
                continue

            paths = image_files(class_path)
            counts[(split, class_name)] = len(paths)
            file_records.extend((path, split, class_name) for path in paths)
            class_formats = Counter()
            class_dimensions = Counter()
            readable_paths = []

            for image_path in paths:
                image_format, image_size, error = inspect_image(image_path)
                if error:
                    unreadable.append((image_path, error))
                    continue
                class_formats[image_format] += 1
                class_dimensions[image_size] += 1
                formats[image_format] += 1
                dimensions[image_size] += 1
                readable_paths.append(image_path)

            print(f"  {class_name}: {len(paths)} file(s), {len(readable_paths)} readable, "
                  f"{len(paths) - len(readable_paths)} unreadable")
            print(f"    Formats: {format_counts_text(class_formats)}")
            print(f"    Dimensions: {dimension_counts_text(class_dimensions)}")

            if display_samples and class_name not in {name for name, _ in samples}:
                samples.extend((class_name, path) for path in readable_paths[:SAMPLES_PER_CLASS])

    duplicates = find_exact_duplicates(file_records)
    split_counts = {split: Counter() for split in SPLITS}
    for (split, class_name), count in counts.items():
        split_counts[split][class_name] = count

    print("\n=== Summary ===")
    print("Image files by split/class:")
    for split in SPLITS:
        print(f"  {split}: " + ", ".join(
            f"{class_name}={counts.get((split, class_name), 'folder missing')}" for class_name in CLASSES
        ))
        print(f"    Class balance: {class_balance_text(split_counts[split])}")
    all_counts = Counter()
    for values in split_counts.values():
        all_counts.update(values)
    print(f"Overall class balance: {class_balance_text(all_counts)}")
    print(f"Readable image formats: {format_counts_text(formats)}")
    print(f"Readable image dimensions: {dimension_counts_text(dimensions)}")
    print(f"Unreadable/corrupted images: {len(unreadable)} (no files were changed)")
    if missing_folders:
        print("Missing expected folders:")
        for folder in missing_folders:
            print(f"  - {folder}")
    else:
        print("All expected split and class folders are present.")
    if unreadable:
        for image_path, error in unreadable:
            print(f"  - {image_path}: {error}")
    print(f"Exact duplicate groups: {len(duplicates)} (byte-for-byte SHA-256 match)")
    cross_split_duplicates = sum(group["cross_split"] for group in duplicates)
    print(f"Exact duplicate groups spanning different splits: {cross_split_duplicates}")
    for group in duplicates:
        split_names = sorted({split for _, split, _ in group['entries']})
        cross_label = "across splits" if group["cross_split"] else "within a split"
        print(f"  - {len(group['entries'])} copies {cross_label}; splits={', '.join(split_names)}")
        for image_path, split, class_name in group["entries"]:
            print(f"      {image_path.relative_to(dataset_root).as_posix()}")

    report = write_report(
        dataset_root, counts, split_counts, formats, dimensions,
        missing_folders, unreadable, duplicates,
    )
    print(f"\nMarkdown report written to: {report}")

    if display_samples:
        show_samples(samples)
    return report


def write_report(root, counts, split_counts, formats, dimensions, missing, unreadable, duplicates):
    report_path = Path(__file__).with_name("dataset_report.md")
    total_counts = Counter()
    for values in split_counts.values():
        total_counts.update(values)
    lines = [
        "# X-Ray Vision dataset report",
        "",
        "Generated by `inspect_data.py` from the KaggleHub dataset copy available on the machine that ran the script.",
        "The original dataset was read only. No images were moved, deleted, or modified, and no split was created.",
        "",
        "## Existing split counts",
        "",
        "| Split | NORMAL | PNEUMONIA | Total |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        normal = counts.get((split, "NORMAL"), "missing folder")
        pneumonia = counts.get((split, "PNEUMONIA"), "missing folder")
        total = normal + pneumonia if isinstance(normal, int) and isinstance(pneumonia, int) else "incomplete"
        lines.append(f"| {split} | {normal} | {pneumonia} | {total} |")
    lines.extend(["", "## Class balance", ""])
    for split in SPLITS:
        lines.append(f"- **{split}:** {class_balance_text(split_counts[split])}")
    lines.append(f"- **Overall:** {class_balance_text(total_counts)}")
    lines.extend(["", "## Image properties", "", f"- **Formats (readable images):** {format_counts_text(formats)}",
                  f"- **Dimensions (readable images):** {dimension_counts_text(dimensions, limit=20)}",
                  f"- **Unreadable/corrupted images:** {len(unreadable)}"])
    if unreadable:
        lines.extend(["", "Unreadable files:"])
        lines.extend(f"- `{path.relative_to(root).as_posix()}` — {error}" for path, error in unreadable)
    lines.extend(["", "## Exact duplicates", "",
                  f"Found **{len(duplicates)} exact duplicate group(s)** using SHA-256 of file bytes. "
                  f"**{sum(group['cross_split'] for group in duplicates)} group(s) span different splits.** "
                  "This does not detect visually identical images saved with different encodings."])
    if duplicates:
        for group in duplicates:
            scope = "across splits" if group["cross_split"] else "within a split"
            lines.append(f"- {len(group['entries'])} copies ({scope}):")
            lines.extend(
                f"  - `{split}/{class_name}/{path.relative_to(root / split / class_name).as_posix()}`"
                for path, split, class_name in group["entries"]
            )
    else:
        lines.append("- None found.")
    lines.extend(["", "## Folder checks", ""])
    if missing:
        lines.append("Missing expected folders:")
        lines.extend(f"- `{folder.relative_to(root).as_posix()}`" for folder in missing)
    else:
        lines.append("All expected `train`, `val`, `test` and `NORMAL`, `PNEUMONIA` folders are present.")
    lines.extend([
        "",
        "## Preparation notes",
        "",
        "- Class labels in `data_pipeline.py`: `NORMAL = 0`, `PNEUMONIA = 1`.",
        "- Initial loader behavior: RGB conversion, deterministic bilinear resize (default 224 × 224), and float tensors scaled to [0, 1]. No random augmentation is applied.",
        "- The loader exposes train and validation by default; the test loader is opt-in for final evaluation only.",
        "- Duplicate findings are reports only. The existing splits are retained unchanged.",
        "- Team should review any cross-split duplicates before model evaluation; do not alter splits without agreement.",
        "- Local Windows loader verification was blocked by an Application Control policy preventing PyTorch's native `torch._C` DLL from loading. The recorded Google Colab verification produced image shape `[4, 3, 224, 224]`, `float32` tensors with values in `[0, 1]`, and label shape `[4]`; that run exposed only `train` and `val` by default, with test loading opt-in and not verified there.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Inspect existing chest X-ray splits and write a report.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="optional local dataset directory; by default KaggleHub resolves/downloads it",
    )
    parser.add_argument("--show-samples", action="store_true", help="open a few readable images per class")
    args = parser.parse_args()

    if args.dataset_dir:
        dataset_root = resolve_dataset_root(args.dataset_dir)
        print(f"Using supplied dataset path: {dataset_root}")
    else:
        dataset_root = download_dataset()
    inspect_dataset(dataset_root, display_samples=args.show_samples)


if __name__ == "__main__":
    main()
