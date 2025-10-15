"""Compatibility layer re-exporting the split image helpers."""
from .image_utils import split_with_fixed_position
from .split_image.dialog import CropHandle, ManualSplitDialog
from .split_image.entries import (
    ManualSplitEntry,
    collect_resolution_metrics,
    enforce_entry_targets,
)
from .split_image.processing import (
    initSplitImage,
    parseImage,
    trim_page_to_resolution,
    _fit_page_to_canvas,
)

__all__ = [
    "CropHandle",
    "ManualSplitDialog",
    "ManualSplitEntry",
    "collect_resolution_metrics",
    "enforce_entry_targets",
    "initSplitImage",
    "parseImage",
    "trim_page_to_resolution",
    "_fit_page_to_canvas",
    "split_with_fixed_position",
]
