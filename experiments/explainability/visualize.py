"""Helpers for saving grayscale X-ray overlays and review contact sheets."""

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image


def save_overlay(image_tensor, heatmap, destination, alpha=0.38):
    """Save a readable heatmap over the already-preprocessed grayscale X-ray."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    image = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    gray = np.clip(image.mean(axis=2), 0, 1)
    cam = heatmap.detach().cpu().float().numpy()
    if cam.shape != gray.shape:
        raise ValueError(f"Heatmap shape {cam.shape} does not match image shape {gray.shape}")
    color = colormaps["jet"](np.clip(cam, 0, 1))[..., :3]
    heat_alpha = alpha * np.clip(cam, 0, 1)[..., None]
    overlay = np.clip((1 - heat_alpha) * gray[..., None] + heat_alpha * color, 0, 1)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(overlay * 255).astype(np.uint8), mode="RGB").save(destination)
    return destination


def save_contact_sheet(records, destination, columns=4):
    """Write a labeled grid of successfully generated overlays."""
    from matplotlib import pyplot as plt

    images = [record for record in records if record.get("overlay_path")]
    if not images:
        return None
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis, record in zip(axes.flat, images):
        with Image.open(record["overlay_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"true={record['true_label']} pred={record['predicted_label']}\n{Path(record['image_path']).name}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(images):]:
        axis.axis("off")
    figure.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination

