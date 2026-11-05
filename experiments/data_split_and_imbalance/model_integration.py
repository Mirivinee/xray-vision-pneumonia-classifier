"""Adapter from the experiment scripts to Suraj's existing scratch CNN."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_scratch.scratch_cnn import ScratchCNN


MODEL_NAME = "cnn_scratch.ScratchCNN"
INPUT_SHAPE = (3, 224, 224)
LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
OUTPUT_CONTRACT = "one raw PNEUMONIA logit per image, shape [B]"


def build_model(dropout=0.5):
    """Build the actual teammate model; no duplicate architecture is defined here."""
    return ScratchCNN(dropout=dropout, in_channels=3)
