import os
from pathlib import Path

import cv2
import numpy as np

from .image_utils import crop_to_content, iter_image_files, load_image, save_with_dpi, split_spread


def initSplitImage(self):
    source_dir = Path(self.fileurl)
    destination_dir = Path(self.directoryName)
    destination_dir.mkdir(parents=True, exist_ok=True)

    files = iter_image_files(source_dir)
    total = len(files)
    processed = 0

    if not files:
        return

    print("__СТАРТ РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ__")

    for file_path in files:
        result = self.parseImage(file_path)
        if result:
            processed += 1
            self.proc.emit(int(processed * 100 / total))


def _fit_page_to_canvas(page: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize *page* proportionally to fit inside the target canvas."""

    if page.ndim == 2:
        page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    if page.shape[0] == target_height and page.shape[1] == target_width:
        return page

    scale = min(target_width / page.shape[1], target_height / page.shape[0])
    scale = max(scale, 1e-6)

    new_width = max(1, int(round(page.shape[1] * scale)))
    new_height = max(1, int(round(page.shape[0] * scale)))

    interpolation = cv2.INTER_LANCZOS4 if scale >= 1 else cv2.INTER_AREA
    resized = cv2.resize(page, (new_width, new_height), interpolation=interpolation)

    channels = 1 if resized.ndim == 2 else resized.shape[2]
    canvas = np.zeros((target_height, target_width, channels), dtype=resized.dtype)

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def parseImage(self, file_path: Path) -> str | None:
    file_path = Path(file_path)
    relative = Path(os.path.relpath(file_path, self.fileurl))
    target_dir = Path(self.directoryName) / relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(file_path)

    if getattr(self, "isRemoveBorder", False):
        pad = self.border_px if getattr(self, "isAddBorder", False) else 0
        cropped, _ = crop_to_content(image, pad_x=pad, pad_y=pad)
        if cropped is not None:
            image = cropped

    height, width = image.shape[:2]

    if height >= width:
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    try:
        spread = split_spread(image, self.width_px, self.pxMediumVal)
    except Exception as exc:
        print(f"[WARN] Не удалось разделить {file_path}: {exc}")
        save_path = target_dir / relative.name
        save_with_dpi(image, save_path, self.dpi)
        return str(save_path)

    number = relative.stem
    left_name = f"{number}_1.jpg"
    right_name = f"{number}_2.jpg"

    left_path = target_dir / left_name
    right_path = target_dir / right_name

    for page_image, page_path in ((spread.left, left_path), (spread.right, right_path)):
        if self.isPxIdentically:
            if self.width_img and self.height_img:
                page_image = _fit_page_to_canvas(page_image, self.width_img, self.height_img)
            else:
                self.width_img, self.height_img = page_image.shape[1], page_image.shape[0]

        save_with_dpi(page_image, page_path, self.dpi)

    return str(left_path)
