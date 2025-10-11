"""Remove noisy black borders from scans."""
from __future__ import annotations

import os
from pathlib import Path

import cv2

from .image_utils import apply_border, iter_image_files, load_image, save_with_dpi


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
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # If the detected area is suspiciously small, keep the original image.
    if w / width < self.kf_w or h / height < self.kf_h:
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    if self.isAddBorder:
        padded = apply_border(image[y : y + h, x : x + w], self.border_px)
    else:
        padded = image[y : y + h, x : x + w]

    save_path = target_dir / relative.name
    save_with_dpi(padded, save_path, self.dpi)
    return str(save_path)
