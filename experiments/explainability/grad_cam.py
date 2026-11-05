"""Shared Grad-CAM calculation for convolutional binary classifiers."""

import torch
import torch.nn.functional as F
from torch import nn

from .model_adapters import extract_pneumonia_logit


def compute_grad_cam(model, inputs, target_layer):
    """Return normalized pneumonia-score CAMs and logits, each with one row per image."""
    if not isinstance(target_layer, nn.Conv2d) or not any(
        module is target_layer for module in model.modules()
    ):
        raise ValueError("Grad-CAM target layer must be a convolution contained in the model")
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError(f"Expected RGB input [B,3,H,W]; got {tuple(inputs.shape)}")
    model.eval()
    captured = {}

    def save_activation(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Target convolution must produce a [B,C,H,W] activation tensor")
        captured["activation"] = output

    handle = target_layer.register_forward_hook(save_activation)
    try:
        with torch.enable_grad():
            logits = model(inputs)
            scores = extract_pneumonia_logit(logits)
            if scores.shape[0] != inputs.shape[0]:
                raise ValueError("Model output batch size does not match the input batch size")
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError("The selected target layer did not produce activations")
            try:
                gradients = torch.autograd.grad(
                    scores.sum(), activation, retain_graph=False, allow_unused=True
                )[0]
            except RuntimeError as error:
                raise RuntimeError("Could not compute gradients for the selected Grad-CAM layer") from error
            if gradients is None:
                raise RuntimeError("No gradients reached the selected Grad-CAM layer")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            flat = cam.flatten(start_dim=1)
            maxima = flat.max(dim=1).values.view(-1, 1, 1, 1)
            cam = torch.where(maxima > 0, cam / maxima.clamp_min(torch.finfo(cam.dtype).eps), torch.zeros_like(cam))
            if not torch.isfinite(cam).all():
                raise RuntimeError("Grad-CAM produced non-finite values")
            return cam[:, 0].detach(), scores.detach()
    finally:
        handle.remove()

