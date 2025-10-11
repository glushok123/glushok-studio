"""Utility helpers for image IO and geometry shared between modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import os

import cv2
import numpy as np
from PIL import Image


DEFAULT_JPEG_QUALITY = 95


def ensure_directory(path: Path) -> None:
    """Create *path* (directory) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(root: Path) -> list[Path]:
    """Return a flat list of files inside *root* (recursively)."""
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            files.append(base / filename)
    return files


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from *path* using ``cv2.imdecode``.

    The helper mirrors the old ``np.fromfile`` + ``cv2.imdecode`` approach so
    that Windows paths with Cyrillic characters work correctly.  The image is
    always returned in BGR format with 3 channels.
    """
    file_path = Path(path)
    data = np.fromfile(str(file_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to decode image: {file_path}")

    if image.ndim == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def to_pillow(image: np.ndarray) -> Image.Image:
    """Convert BGR/GRAY numpy array to a :class:`PIL.Image.Image`."""
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def save_with_dpi(image: np.ndarray, destination: str | Path, dpi: int) -> None:
    """Persist *image* as JPEG with the given *dpi* metadata."""
    dest_path = Path(destination)
    ensure_directory(dest_path.parent)
    pil_image = to_pillow(image)
    pil_image.save(dest_path, format="JPEG", dpi=(dpi, dpi), quality=DEFAULT_JPEG_QUALITY)


def apply_border(image: np.ndarray, border_px: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Return *image* with a constant border on all sides."""
    if border_px <= 0:
        return image
    return cv2.copyMakeBorder(
        image,
        top=border_px,
        bottom=border_px,
        left=border_px,
        right=border_px,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )


@dataclass(frozen=True)
class SplitResult:
    left: np.ndarray
    right: np.ndarray
    split_x: int


def find_split_column(gray: np.ndarray, window_ratio: float = 0.08) -> int:
    """Detect the most probable gutter position for a spread ``gray`` image.

    The algorithm projects pixel intensities along the X axis and searches for
    the lightest column close to the centre.  It proved to be far more stable
    than the old custom heuristics and never returns an index outside the image
    bounds.
    """
    height, width = gray.shape
    projection = gray.mean(axis=0)
    window = max(3, int(width * window_ratio) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(projection, kernel, mode="same")

    centre = width // 2
    radius = max(width // 6, window)
    start = max(0, centre - radius)
    end = min(width, centre + radius)
    if start >= end:
        return centre

    slice_projection = smoothed[start:end]
    local_index = int(np.argmin(slice_projection))
    split_x = start + local_index
    return int(np.clip(split_x, 0, width - 1))


def split_spread(image: np.ndarray, overlap: int) -> SplitResult:
    """Split a double-page spread *image* into two halves with *overlap* pixels.

    The function keeps the pages symmetric: both parts contain the shared
    gutter area, so the user can later decide what to keep when performing
    manual post-processing.
    """
    if image.ndim == 2:
        gray = image
        colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        colour = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    split_x = find_split_column(gray)

    overlap = max(0, int(overlap))
    left_end = min(width, split_x + overlap)
    right_start = max(0, split_x - overlap)

    left_page = colour[:, :left_end]
    right_page = colour[:, right_start:]

    if left_page.size == 0 or right_page.size == 0:
        half = width // 2
        left_page = colour[:, :half]
        right_page = colour[:, half:]
        split_x = half

    return SplitResult(left=left_page, right=right_page, split_x=split_x)
