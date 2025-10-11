"""Remove noisy black borders from scans."""
from __future__ import annotations

import os
from pathlib import Path

import cv2

from .image_utils import (
    crop_to_content,
    detect_content_bounds,
    iter_image_files,
    load_image,
    save_with_dpi,
)


def initRemoveBorder(self):
    source_dir = Path(self.fileurl)
    destination_dir = Path(self.directoryName + self.postfix)
    destination_dir.mkdir(parents=True, exist_ok=True)

    files = iter_image_files(source_dir)
    total = len(files)
    processed = 0

    print(f"__СТАРТ УДАЛЕНИЯ РАМКИ__{total}")

    for file_path in files:
        result = self.removeBorder(file_path)
        if result:
            processed += 1
            self.proc.emit(int(processed * 100 / max(total, 1)))


def removeBorder(self, file_path: Path) -> str | None:
    file_path = Path(file_path)
    relative = Path(os.path.relpath(file_path, self.fileurl))
    target_dir = Path(self.directoryName + self.postfix) / relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = load_image(file_path)
    except ValueError:
        return None

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bounds = detect_content_bounds(gray)
    if bounds is None:
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    # If the detected area is suspiciously small, keep the original image.
    if bounds.width / width < self.kf_w or bounds.height / height < self.kf_h:
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    pad_x = self.border_px if self.isAddBorder else 0
    pad_y = self.border_px if self.isAddBorder else 0

    cropped, _ = crop_to_content(image, pad_x=pad_x, pad_y=pad_y, bounds=bounds)
    if cropped is None:
        cropped = image

    save_path = target_dir / relative.name
    save_with_dpi(cropped, save_path, self.dpi)
    return str(save_path)
