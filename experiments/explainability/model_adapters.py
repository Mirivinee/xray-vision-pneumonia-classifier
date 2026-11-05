"""Model construction, checkpoint loading, and binary-output adapters."""

from pathlib import Path

import torch
from torch import nn

from cnn_scratch.scratch_cnn import ScratchCNN


ARCHITECTURES = ("scratch_cnn", "resnet50", "densenet121")
LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}


def build_model(architecture, dropout=0.5):
    """Build an architecture without downloading weights or contacting a server."""
    if architecture == "scratch_cnn":
        return ScratchCNN(dropout=dropout, in_channels=3)
    if architecture == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if architecture == "densenet121":
        from torchvision.models import densenet121

        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model
    raise ValueError(f"Unsupported architecture {architecture!r}; choose {', '.join(ARCHITECTURES)}")


def select_target_layer(model, architecture):
    """Find the last convolution in the architecture's final feature stage."""
    if architecture == "scratch_cnn":
        target = getattr(model, "gradcam_target_layer", None)
    else:
        stage_prefix = {"resnet50": "layer4", "densenet121": "features.denseblock4"}.get(architecture)
        if stage_prefix is None:
            raise ValueError(f"Unsupported architecture {architecture!r}")
        candidates = [
            module for name, module in model.named_modules()
            if name.startswith(stage_prefix) and isinstance(module, nn.Conv2d)
        ]
        target = candidates[-1] if candidates else None
    if not isinstance(target, nn.Conv2d):
        raise ValueError(f"No suitable final convolutional target layer found for {architecture}")
    if not any(module is target for module in model.modules()):
        raise ValueError("Grad-CAM target layer is not part of the selected model")
    return target


def extract_pneumonia_logit(logits):
    """Return one pneumonia score per image from a binary model's output.

    ScratchCNN emits one raw pneumonia logit. Two-class models emit columns in
    NORMAL, PNEUMONIA order, so column 1 is the pneumonia logit.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model output must be a torch.Tensor")
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, LABEL_TO_ID["PNEUMONIA"]]
    raise ValueError(f"Expected model output [B], [B,1], or [B,2]; got {tuple(logits.shape)}")


def pneumonia_probability(logits):
    """Convert supported binary logits to P(PNEUMONIA)."""
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, LABEL_TO_ID["PNEUMONIA"]]
    return torch.sigmoid(extract_pneumonia_logit(logits))


def _checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError("Checkpoint must contain a model_state or state_dict mapping")


def load_model_checkpoint(architecture, checkpoint_path, device):
    """Load an existing checkpoint without changing the checkpoint file."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if architecture == "scratch_cnn" and config.get("model") not in (None, "cnn_scratch.ScratchCNN"):
        raise ValueError(f"Checkpoint identifies a different model: {config.get('model')!r}")
    if config.get("label_to_id") not in (None, LABEL_TO_ID):
        raise ValueError("Checkpoint labels must use NORMAL=0 and PNEUMONIA=1")
    model = build_model(architecture, dropout=float(config.get("dropout", 0.5)))
    state = _checkpoint_state(checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint tensors do not match {architecture}; verify architecture and classifier output size. {error}"
        ) from error
    model.to(device).eval()
    target = select_target_layer(model, architecture)
    return model, target, metadata


def resolve_normalization(requested, checkpoint_metadata):
    """Choose normalization explicitly or infer only a recognized checkpoint setting."""
    if requested in ("none", "imagenet"):
        return requested
    config = checkpoint_metadata.get("config", {}) if isinstance(checkpoint_metadata, dict) else {}
    description = str(config.get("preprocessing", "")).lower()
    if "no channel normalization" in description or "[0,1]" in description:
        return "none"
    if "imagenet" in description:
        return "imagenet"
    raise ValueError(
        "Checkpoint preprocessing is not recorded clearly. Pass --normalization none or --normalization imagenet "
        "to match the model's training input."
    )

