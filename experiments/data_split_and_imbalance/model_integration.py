"""Single integration point for the partner's model implementation.

Set PARTNER_MODEL_FACTORY to an import path like models.baseline:build_model,
or pass --model-factory. The factory must accept num_classes=2 and return a
PyTorch module. Input/output expectations are listed in INPUT_CONTRACT/OUTPUT_CONTRACT.
"""

import importlib
import os
import sys
from pathlib import Path


PARTNER_MODEL_FACTORY = None
INPUT_CONTRACT = "RGB float32 [B,3,224,224], values [0,1]"
OUTPUT_CONTRACT = "two logits [B,2], NORMAL=0 then PNEUMONIA=1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_factory_spec(argument=None):
    """Return the CLI/environment/file setting or give a useful handoff error."""
    spec = argument or os.environ.get("XRAY_MODEL_FACTORY") or PARTNER_MODEL_FACTORY
    if not spec:
        raise RuntimeError(
            "Partner model is not configured. Add its module to the project, set "
            "PARTNER_MODEL_FACTORY in model_integration.py, or pass --model-factory "
            "module.path:build_model. The factory must accept num_classes=2."
        )
    if ":" not in spec:
        raise ValueError("Model factory must use module.path:function format")
    return spec


def build_partner_model(factory_spec):
    """Import and construct the partner model with clear error context."""
    module_name, function_name = factory_spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except Exception as error:
        raise RuntimeError(
            f"Could not import model factory {factory_spec!r}. Put the model module "
            "in the project or correct the --model-factory path."
        ) from error
    try:
        model = factory(num_classes=2)
    except Exception as error:
        raise RuntimeError(
            f"Model factory {factory_spec!r} must accept num_classes=2 and return a PyTorch module."
        ) from error
    if model is None:
        raise RuntimeError(f"Model factory {factory_spec!r} returned None.")
    return model
