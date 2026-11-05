"""Optional, deterministic image-size handling for prepared dataset variants."""

from PIL import Image


def resize_image(image, size=None):
    """Resize to ``(width, height)``; with ``size=None`` preserve dimensions."""
    if size is None:
        return image.copy()
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError("size must be (width, height) with positive integers, or None")
    width, height = (int(value) for value in size)
    return image.resize((width, height), Image.Resampling.BILINEAR)

