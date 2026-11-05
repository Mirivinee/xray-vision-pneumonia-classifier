"""Compatibility entry point for the shared preparation pipeline.

Run ``python experiments/data_split_and_imbalance/split_data.py`` as before;
the implementation now lives in ``data/preparation_pipeline.py``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preparation_pipeline import (
    DEFAULT_SEED,
    MANIFEST_DIR,
    RATIOS,
    create_manifests,
    find_exact_duplicates,
    main,
    quarantine_exact_duplicates,
    report_records,
    restore_quarantine,
    scan_images,
    sha256_file,
    split_counts,
    write_audit_report,
)


if __name__ == "__main__":
    main()
