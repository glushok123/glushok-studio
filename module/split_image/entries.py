"""Data structures and helpers for manual spread splitting."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PyQt5.QtGui import QImage

from ..image_utils import load_image


@dataclass
class ManualSplitEntry:
    source_path: Path
    relative: Path
    target_dir: Path
    left_path: Path
    right_path: Path
    auto_split_x: int
    split_x: int
    overlap: int
    image_width: int
    image_height: int
    crop_left: int = 0
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0
    rotation_deg: float = 0.0
    split_disabled: bool = False
    split_x_base: int = field(init=False)
    split_ratio: float = field(init=False)
    _base_image: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _preview_qimage: QImage | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def width(self) -> int:
        return max(1, int(self.image_width))

    @property
    def height(self) -> int:
        return max(1, int(self.image_height))

    def __post_init__(self) -> None:
        self.image_width = max(1, int(self.image_width))
        self.image_height = max(1, int(self.image_height))
        self.crop_left = max(0, int(self.crop_left))
        self.crop_top = max(0, int(self.crop_top))
        self.crop_right = self.crop_right or self.width
        self.crop_bottom = self.crop_bottom or self.height
        self.crop_right = max(self.crop_left + 1, int(self.crop_right))
        self.crop_bottom = max(self.crop_top + 1, int(self.crop_bottom))
        self.rotation_deg = float(self.rotation_deg)
        self.split_disabled = bool(self.split_disabled)
        if self.current_width > 0:
            self.split_x = int(np.clip(self.split_x, 0, self.current_width - 1))
            self.split_ratio = float(np.clip(self.split_x / self.current_width, 0.0, 1.0))
        else:
            self.split_x = 0
            self.split_ratio = 0.5
        self.split_x_base = int(self.crop_left + self.split_x)

    def preview_qimage(self) -> QImage:
        base = self.ensure_loaded()
        if self._preview_qimage is None:
            self._preview_qimage = _numpy_to_qimage(base)
        return self._preview_qimage

    def ensure_loaded(self) -> np.ndarray:
        if self._base_image is None:
            image = load_image(self.source_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {self.source_path}")
            self._base_image = np.ascontiguousarray(image)
            self.image_height, self.image_width = self._base_image.shape[:2]
            self._preview_qimage = None
            if self.crop_right <= self.crop_left:
                self.crop_right = self.width
            if self.crop_bottom <= self.crop_top:
                self.crop_bottom = self.height
            self.set_split_x(int(self.split_x))
        return self._base_image

    def release_image(self) -> None:
        self._base_image = None
        self._preview_qimage = None

    @property
    def current_width(self) -> int:
        return max(1, int(self.crop_right - self.crop_left))

    @property
    def current_height(self) -> int:
        return max(1, int(self.crop_bottom - self.crop_top))

    @property
    def crop_centre(self) -> tuple[float, float]:
        return (
            float(self.crop_left + self.current_width / 2.0),
            float(self.crop_top + self.current_height / 2.0),
        )

    def set_split_x(self, value: int) -> None:
        width = self.current_width
        if width <= 1:
            self.split_x = 0
            self.split_ratio = 0.5
            self.split_x_base = int(self.crop_left)
            return

        value = int(np.clip(value, 0, width - 1))
        self.split_x = value
        self.split_ratio = float(np.clip(value / width, 0.0, 1.0))
        self.split_x_base = int(self.crop_left + self.split_x)

    def set_split_base(self, base: int) -> None:
        width = self.current_width
        if width <= 1:
            self.split_x = 0
            self.split_ratio = 0.5
            self.split_x_base = int(self.crop_left)
            return

        base = int(base)
        value = base - int(self.crop_left)
        value = int(np.clip(value, 0, width - 1))
        self.split_x = value
        self.split_ratio = float(np.clip(value / width, 0.0, 1.0))
        self.split_x_base = int(self.crop_left + self.split_x)

    def update_split_from_ratio(self) -> None:
        width = self.current_width
        if width <= 1:
            self.split_x = 0
            self.split_x_base = int(self.crop_left)
            return

        ratio = float(np.clip(self.split_ratio, 0.0, 1.0))
        value = int(round(ratio * (width - 1)))
        value = int(np.clip(value, 0, width - 1))
        self.split_x = value
        self.split_x_base = int(self.crop_left + self.split_x)

    def build_processed_image(self) -> np.ndarray:
        base = self.ensure_loaded()
        left = max(0, min(int(self.crop_left), base.shape[1] - 1))
        right = max(left + 1, min(int(self.crop_right), base.shape[1]))
        top = max(0, min(int(self.crop_top), base.shape[0] - 1))
        bottom = max(top + 1, min(int(self.crop_bottom), base.shape[0]))

        if right <= left or bottom <= top:
            return base.copy()

        if abs(self.rotation_deg) < 1e-3:
            return base[top:bottom, left:right].copy()

        centre_x, centre_y = self.crop_centre
        centre = (float(centre_x), float(centre_y))

        matrix = cv2.getRotationMatrix2D(centre, -self.rotation_deg, 1.0)

        height, width = base.shape[:2]
        corners = np.array(
            [
                [0.0, 0.0, 1.0],
                [float(width), 0.0, 1.0],
                [float(width), float(height), 1.0],
                [0.0, float(height), 1.0],
            ],
            dtype=np.float32,
        )
        rotated_corners = (matrix @ corners.T).T
        min_x = float(rotated_corners[:, 0].min())
        max_x = float(rotated_corners[:, 0].max())
        min_y = float(rotated_corners[:, 1].min())
        max_y = float(rotated_corners[:, 1].max())

        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))
        if new_width <= 0 or new_height <= 0:
            return base[top:bottom, left:right].copy()

        matrix[0, 2] -= min_x
        matrix[1, 2] -= min_y

        rotated = cv2.warpAffine(
            base,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        crop_points = np.array(
            [
                [float(left), float(top), 1.0],
                [float(right), float(top), 1.0],
                [float(right), float(bottom), 1.0],
                [float(left), float(bottom), 1.0],
            ],
            dtype=np.float32,
        )
        rotated_crop = (matrix @ crop_points.T).T

        crop_min_x = max(0, int(np.floor(rotated_crop[:, 0].min())))
        crop_max_x = min(new_width, int(np.ceil(rotated_crop[:, 0].max())))
        crop_min_y = max(0, int(np.floor(rotated_crop[:, 1].min())))
        crop_max_y = min(new_height, int(np.ceil(rotated_crop[:, 1].max())))

        if crop_max_x <= crop_min_x or crop_max_y <= crop_min_y:
            return rotated

        cropped = rotated[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
        if not cropped.size:
            return rotated

        polygon = rotated_crop[:, :2] - np.array(
            [float(crop_min_x), float(crop_min_y)], dtype=np.float32
        )
        mask_height = crop_max_y - crop_min_y
        mask_width = crop_max_x - crop_min_x
        mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        if mask_width <= 0 or mask_height <= 0:
            return cropped

        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, float(mask_width - 1))
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, float(mask_height - 1))
        cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return cropped

        x, y, w_box, h_box = cv2.boundingRect(coords)
        trimmed = cropped[y : y + h_box, x : x + w_box]
        return trimmed if trimmed.size else cropped

    def final_split_position(self, final_width: int) -> int:
        if final_width <= 1:
            return 0

        ratio = float(np.clip(self.split_ratio, 0.0, 1.0))
        position = int(round(ratio * (final_width - 1)))
        return int(np.clip(position, 0, final_width - 1))


def _entry_page_dimensions(entry: ManualSplitEntry) -> dict[str, int]:
    overlap = max(0, int(entry.overlap))
    base = int(entry.split_x_base)

    height = entry.current_height
    width = entry.current_width

    if entry.split_disabled:
        return {
            "height": height,
            "left_width": 0,
            "right_width": 0,
            "spread_width": width,
            "spread_height": height,
        }

    left_width = max(0, base - entry.crop_left + overlap)
    right_width = max(0, entry.crop_right - base + overlap)

    return {
        "height": height,
        "left_width": left_width,
        "right_width": right_width,
        "spread_width": 0,
        "spread_height": 0,
    }


def collect_resolution_metrics(entries: List[ManualSplitEntry]) -> dict[str, int | str | None]:
    metrics: dict[str, int | str | None] = {
        "max_height": 0,
        "max_height_index": None,
        "max_left_width": 0,
        "max_left_index": None,
        "max_right_width": 0,
        "max_right_index": None,
        "max_spread_width": 0,
        "max_spread_height": 0,
        "target_page_width": 0,
        "target_page_height": 0,
        "max_width_index": None,
        "max_width_side": None,
        "total_entries": len(entries),
    }

    for idx, entry in enumerate(entries):
        dims = _entry_page_dimensions(entry)

        if dims["height"] > metrics["max_height"]:
            metrics["max_height"] = dims["height"]
            metrics["max_height_index"] = idx

        if not entry.split_disabled:
            if dims["left_width"] > metrics["max_left_width"]:
                metrics["max_left_width"] = dims["left_width"]
                metrics["max_left_index"] = idx
            if dims["right_width"] > metrics["max_right_width"]:
                metrics["max_right_width"] = dims["right_width"]
                metrics["max_right_index"] = idx
        else:
            metrics["max_spread_width"] = max(metrics["max_spread_width"], dims["spread_width"])
            metrics["max_spread_height"] = max(metrics["max_spread_height"], dims["spread_height"])

    metrics["target_page_height"] = metrics["max_height"]
    metrics["target_page_width"] = max(metrics["max_left_width"], metrics["max_right_width"])

    if metrics["max_right_width"] and (
        metrics["max_right_width"] >= metrics["max_left_width"]
    ):
        metrics["max_width_index"] = metrics["max_right_index"]
        metrics["max_width_side"] = "right"
    elif metrics["max_left_width"]:
        metrics["max_width_index"] = metrics["max_left_index"]
        metrics["max_width_side"] = "left"

    return metrics


def _enforce_entry_height(entry: ManualSplitEntry, target_height: int) -> bool:
    target_height = int(target_height)
    if target_height <= 0:
        return False

    max_height = max(1, int(entry.height))
    target_height = min(target_height, max_height)
    if target_height <= 0:
        return False

    current_height = entry.current_height
    if current_height == target_height:
        return False

    centre = entry.crop_top + current_height / 2.0
    new_top = int(round(centre - target_height / 2.0))
    new_bottom = new_top + target_height

    if new_top < 0:
        new_top = 0
        new_bottom = target_height
    if new_bottom > max_height:
        new_bottom = max_height
        new_top = max(0, max_height - target_height)

    if new_bottom - new_top > target_height:
        excess = (new_bottom - new_top) - target_height
        new_top += excess // 2
        new_bottom = new_top + target_height

    if new_bottom - new_top < target_height:
        new_top = max(0, min(new_top, max_height - target_height))
        new_bottom = new_top + target_height

    new_top = max(0, min(new_top, max_height - 1))
    new_bottom = max(new_top + 1, min(new_bottom, max_height))

    if new_bottom - new_top != target_height and target_height == max_height:
        new_top = 0
        new_bottom = max_height

    changed = new_top != entry.crop_top or new_bottom != entry.crop_bottom
    entry.crop_top = new_top
    entry.crop_bottom = new_bottom
    return changed


def _enforce_spread_width(entry: ManualSplitEntry, target_width: int) -> bool:
    target_width = int(target_width)
    if target_width <= 0:
        return False

    max_width = max(1, int(entry.width))
    target_width = min(target_width, max_width)
    if target_width <= 0:
        return False

    current_width = entry.current_width
    if current_width == target_width:
        return False

    centre = entry.crop_left + current_width / 2.0
    new_left = int(round(centre - target_width / 2.0))
    new_right = new_left + target_width

    if new_left < 0:
        new_left = 0
        new_right = target_width
    if new_right > max_width:
        new_right = max_width
        new_left = max(0, max_width - target_width)

    if new_right - new_left > target_width:
        excess = (new_right - new_left) - target_width
        new_left += excess // 2
        new_right = new_left + target_width

    if new_right - new_left < target_width:
        new_left = max(0, min(new_left, max_width - target_width))
        new_right = new_left + target_width

    new_left = max(0, min(new_left, max_width - 1))
    new_right = max(new_left + 1, min(new_right, max_width))

    if new_right - new_left != target_width and target_width == max_width:
        new_left = 0
        new_right = max_width

    changed = new_left != entry.crop_left or new_right != entry.crop_right
    entry.crop_left = new_left
    entry.crop_right = new_right
    return changed


def _enforce_entry_width(entry: ManualSplitEntry, target_width: int) -> bool:
    target_width = int(target_width)
    if target_width <= 0 or entry.split_disabled:
        return False

    base = int(entry.split_x_base)
    max_width = max(1, int(entry.width))
    if base <= 0 or base >= max_width:
        return False

    target_width = min(target_width, max_width)
    if target_width <= 0:
        return False

    min_overlap = max(0, target_width - base, base + target_width - max_width)
    if min_overlap >= target_width:
        return False

    desired_overlap = int(min(target_width - 1, max(min_overlap, int(entry.overlap))))
    new_left = base + desired_overlap - target_width
    new_right = base - desired_overlap + target_width

    if new_left < 0 or new_right > max_width:
        return False

    new_left = int(new_left)
    new_right = int(new_right)

    if new_right <= new_left:
        return False

    changed = (
        new_left != entry.crop_left
        or new_right != entry.crop_right
        or desired_overlap != int(entry.overlap)
    )

    entry.crop_left = new_left
    entry.crop_right = new_right
    entry.overlap = desired_overlap
    entry.set_split_base(base)
    return changed


def enforce_entry_targets(
    entry: ManualSplitEntry,
    *,
    target_page_width: int,
    target_page_height: int,
    target_spread_width: int,
    target_spread_height: int,
) -> None:
    effective_height = target_page_height
    if entry.split_disabled and target_spread_height > 0:
        effective_height = max(effective_height, target_spread_height)

    _enforce_entry_height(entry, effective_height)

    if entry.split_disabled:
        if target_spread_width > 0:
            _enforce_spread_width(entry, target_spread_width)
        return

    _enforce_entry_width(entry, target_page_width)


def _numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        return QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


__all__ = [
    "ManualSplitEntry",
    "collect_resolution_metrics",
    "enforce_entry_targets",
    "_numpy_to_qimage",
]
