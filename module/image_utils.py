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
class BoundingBox:
    """Axis-aligned rectangle represented by its corner coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def expand(self, image_shape: tuple[int, ...], pad_x: int, pad_y: int) -> "BoundingBox":
        """Return a new box enlarged by *pad_x*/*pad_y* pixels and clamped to the image."""

        height, width = image_shape[:2]
        pad_x = max(0, int(pad_x))
        pad_y = max(0, int(pad_y))

        left = max(0, self.left - pad_x)
        top = max(0, self.top - pad_y)
        right = min(width, self.right + pad_x)
        bottom = min(height, self.bottom + pad_y)

        if left >= right or top >= bottom:
            return self

        return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def detect_content_bounds(gray: np.ndarray) -> BoundingBox | None:
    """Detect the main document bounds within the grayscale image *gray*."""

    if gray.ndim != 2:
        raise ValueError("detect_content_bounds expects a single channel image")

    height, width = gray.shape
    image_area = float(width * height)
    if image_area == 0:
        return None

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heuristic: if most of the page ended up black (e.g. the scan is dark),
    # invert the mask so that the document becomes the bright region.
    white_ratio = cv2.countNonZero(thresh) / image_area
    if white_ratio < 0.1:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = image_area * 0.01
    boxes: list[BoundingBox] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(BoundingBox(left=x, top=y, right=x + w, bottom=y + h))

    if not boxes:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(BoundingBox(left=x, top=y, right=x + w, bottom=y + h))

    left = min(box.left for box in boxes)
    top = min(box.top for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)

    left = int(np.clip(left, 0, width))
    top = int(np.clip(top, 0, height))
    right = int(np.clip(right, left, width))
    bottom = int(np.clip(bottom, top, height))

    if right - left <= 0 or bottom - top <= 0:
        return None

    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def crop_to_content(
    image: np.ndarray,
    pad_x: int = 0,
    pad_y: int = 0,
    bounds: BoundingBox | None = None,
) -> tuple[np.ndarray, BoundingBox] | tuple[None, None]:
    """Crop *image* to the detected document bounds and return the crop."""

    if bounds is None:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        bounds = detect_content_bounds(gray)
    if bounds is None:
        return None, None

    expanded = bounds.expand(image.shape, pad_x, pad_y)
    if expanded.width <= 0 or expanded.height <= 0:
        return None, None

    cropped = image[expanded.top : expanded.bottom, expanded.left : expanded.right]
    return cropped, expanded


@dataclass(frozen=True)
class SplitResult:
    left: np.ndarray
    right: np.ndarray
    split_x: int


def _project_text_density(gray: np.ndarray) -> np.ndarray:
    """Return text density projection for each column of ``gray`` image.

    The helper binarises the image and counts how many dark (text) pixels are in
    each column.  Low values indicate blank space such as the gutter.
    """

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = cv2.countNonZero(binary) / max(1, gray.size)
    if white_ratio < 0.5:
        binary = cv2.bitwise_not(binary)

    text_mask = 255 - binary
    projection = text_mask.mean(axis=0)
    return projection


def find_split_column(
    gray: np.ndarray,
    window_ratio: float = 0.08,
    search_radius: int | None = None,
) -> tuple[int, bool]:
    """Detect the most probable gutter position for a spread ``gray`` image.

    Returns a tuple ``(position, is_confident)``.  When ``is_confident`` is
    ``False`` the caller should ignore the detected position and fall back to a
    neutral split (for example the physical centre of the image).
    """

    _, width = gray.shape
    projection = _project_text_density(gray)

    window = max(3, int(width * window_ratio) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(projection, kernel, mode="same")

    centre = width // 2
    if search_radius is None or search_radius <= 0:
        radius = max(width // 6, window)
    else:
        radius = max(window, int(search_radius))

    start = max(0, centre - radius)
    end = min(width, centre + radius)
    if end - start <= 1:
        return centre, False

    slice_projection = smoothed[start:end]
    local_index = int(np.argmin(slice_projection))
    split_x = start + local_index

    local_min = float(slice_projection[local_index])
    local_mean = float(slice_projection.mean())
    local_std = float(slice_projection.std())

    deviation = local_mean - local_min
    relative_drop = deviation / (local_mean + 1e-6)
    z_score = deviation / (local_std + 1e-6) if local_std > 1e-6 else 0.0

    is_confident = (
        local_min <= 5.0
        or relative_drop >= 0.3
        or z_score >= 1.0
    )

    return int(np.clip(split_x, 0, width - 1)), bool(is_confident)


def _build_split_result(colour: np.ndarray, split_x: int, overlap: int) -> SplitResult:
    """Create a :class:`SplitResult` using a fixed ``split_x`` position."""

    height, width = colour.shape[:2]
    split_x = int(np.clip(split_x, 0, width - 1))

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

    left_width = left_page.shape[1]
    right_width = right_page.shape[1]
    target_width = min(left_width, right_width)

    if target_width > 0:
        if left_width > target_width:
            left_page = left_page[:, left_width - target_width :]
        if right_width > target_width:
            right_page = right_page[:, :target_width]

    return SplitResult(left=left_page, right=right_page, split_x=split_x)


def split_spread(image: np.ndarray, overlap: int, centre_tolerance: int = 0) -> SplitResult:
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

    search_radius = max(0, int(centre_tolerance)) if centre_tolerance else None

    bounds = detect_content_bounds(gray)
    split_x = width // 2
    has_gutter = False

    if bounds and bounds.width > 0 and bounds.height > 0:
        crop = gray[bounds.top : bounds.bottom, bounds.left : bounds.right]
        if crop.size:
            local_split, has_gutter = find_split_column(crop, search_radius=search_radius)
            split_x = bounds.left + local_split
    if not has_gutter:
        split_x_candidate, has_gutter = find_split_column(gray, search_radius=search_radius)
        split_x = split_x_candidate if has_gutter else split_x

    centre = width // 2
    tolerance = max(0, int(centre_tolerance))
    if tolerance and abs(split_x - centre) > tolerance:
        has_gutter = False

    if not has_gutter:
        split_x = centre

    return _build_split_result(colour, split_x, overlap)


def split_with_fixed_position(image: np.ndarray, split_x: int, overlap: int) -> SplitResult:
    """Split ``image`` using an explicitly supplied ``split_x`` gutter."""

    if image.ndim == 2:
        colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        colour = image

    return _build_split_result(colour, split_x, overlap)
