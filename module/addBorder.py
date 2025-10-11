"""Add a uniform black frame around processed pages."""
from __future__ import annotations

from pathlib import Path

from .image_utils import apply_border, iter_image_files, load_image, save_with_dpi


def initAddBorder(self):
    source_dir = Path(self.fileurl)
    files = iter_image_files(source_dir)
    self.countFile = len(files)

    if not files:
        return

    for file_path in files:
        self.addBorder(file_path)


def addBorder(self, file_path: Path):
    file_path = Path(file_path)
    image = load_image(file_path)
    bordered = apply_border(image, self.border_px)
    save_with_dpi(bordered, file_path, self.dpi)
