"""A light-weight post border cleaning step used when an artificial frame was added."""
from __future__ import annotations

import os
from pathlib import Path

import cv2

from .image_utils import detect_content_bounds, iter_image_files, load_image, save_with_dpi


def initRemovePostBorder(self):
    source_dir = Path(self.fileurl)
    destination_dir = Path(self.directoryName)
    destination_dir.mkdir(parents=True, exist_ok=True)

    files = iter_image_files(source_dir)
    total = len(files)
    processed = 0

    print("__СТАРТ УДАЛЕНИЯ РАМКИ ДОП__")

    for file_path in files:
        result = self.removePostBorder(file_path)
        if result:
            processed += 1
            self.proc.emit(int(processed * 100 / max(total, 1)))


def removePostBorder(self, file_path: Path) -> str | None:
    file_path = Path(file_path)
    relative = Path(os.path.relpath(file_path, self.fileurl))
    target_dir = Path(self.directoryName) / relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = load_image(file_path)
    except ValueError:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bounds = detect_content_bounds(gray)
    if bounds is None or bounds.width == 0 or bounds.height == 0:
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    pad_x = self.border_px if self.isAddBorder else 0
    pad_y = self.border_px if self.isAddBorder and self.isAddBorderForAll else 0
    expanded = bounds.expand(image.shape, pad_x, pad_y)

    if expanded.width <= 0 or expanded.height <= 0:
        expanded = bounds

    cropped = image[expanded.top : expanded.bottom, expanded.left : expanded.right]

    save_path = target_dir / relative.name
    save_with_dpi(cropped, save_path, self.dpi)
    return str(save_path)
