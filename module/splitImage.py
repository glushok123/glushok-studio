import os
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPen, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .image_utils import (
    crop_to_content,
    iter_image_files,
    load_image,
    save_with_dpi,
    split_spread,
    split_with_fixed_position,
)


@dataclass
class ManualSplitEntry:
    relative: Path
    target_dir: Path
    image: np.ndarray
    left_path: Path
    right_path: Path
    auto_split_x: int
    split_x: int
    overlap: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1]) if self.image.size else 0

    @property
    def height(self) -> int:
        return int(self.image.shape[0]) if self.image.size else 0


def _numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        return QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


class ManualSplitDialog(QDialog):
    def __init__(self, entries: List[ManualSplitEntry], parent=None):
        super().__init__(parent)
        self.entries = entries
        self.current_index = 0
        self._pixmap_item = None
        self._split_line = None
        self._current_pixmap = None

        self.setWindowTitle("Ручная корректировка середины")
        self.resize(900, 600)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)

        self.fileLabel = QLabel()
        self.positionLabel = QLabel()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)

        self.resetButton = QPushButton("Сбросить")
        self.prevButton = QPushButton("Предыдущее")
        self.nextButton = QPushButton("Следующее")
        self.finishButton = QPushButton("Готово")
        self.cancelButton = QPushButton("Отмена")

        self.resetButton.clicked.connect(self.reset_current)
        self.prevButton.clicked.connect(self.goto_previous)
        self.nextButton.clicked.connect(self.goto_next)
        self.finishButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.prevButton)
        buttonLayout.addWidget(self.nextButton)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.resetButton)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.cancelButton)
        buttonLayout.addWidget(self.finishButton)

        layout = QVBoxLayout()
        layout.addWidget(self.fileLabel)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.positionLabel)
        layout.addWidget(self.slider)
        layout.addLayout(buttonLayout)

        self.setLayout(layout)

        if self.entries:
            self.display_current_entry()
        else:
            self.slider.setEnabled(False)
            self.prevButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.resetButton.setEnabled(False)
            self.finishButton.setEnabled(False)

    def display_current_entry(self):
        entry = self.entries[self.current_index]
        self.scene.clear()
        self._pixmap_item = None
        self._split_line = None

        pixmap = QPixmap.fromImage(_numpy_to_qimage(entry.image))
        self._current_pixmap = pixmap
        self._pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        pen = QPen(Qt.red)
        pen.setWidth(max(2, pixmap.width() // 400))
        self._split_line = self.scene.addLine(entry.split_x, 0, entry.split_x, pixmap.height(), pen)

        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.fileLabel.setText(f"{entry.relative.name} ({self.current_index + 1}/{len(self.entries)})")
        self.positionLabel.setText(f"Позиция разреза: {entry.split_x}px")

        maximum = max(1, entry.width - 1)
        self.slider.blockSignals(True)
        self.slider.setMaximum(maximum)
        self.slider.setValue(entry.split_x)
        self.slider.blockSignals(False)

        self.update_navigation()

    def update_navigation(self):
        self.prevButton.setEnabled(self.current_index > 0)
        self.nextButton.setEnabled(self.current_index < len(self.entries) - 1)

    def on_slider_changed(self, value: int):
        entry = self.entries[self.current_index]
        entry.split_x = int(value)
        self.positionLabel.setText(f"Позиция разреза: {entry.split_x}px")
        if self._split_line is not None:
            self._split_line.setLine(entry.split_x, 0, entry.split_x, entry.height)

    def goto_previous(self):
        if self.current_index <= 0:
            return
        self.current_index -= 1
        self.display_current_entry()

    def goto_next(self):
        if self.current_index >= len(self.entries) - 1:
            return
        self.current_index += 1
        self.display_current_entry()

    def reset_current(self):
        entry = self.entries[self.current_index]
        self.slider.blockSignals(True)
        self.slider.setValue(entry.auto_split_x)
        self.slider.blockSignals(False)
        self.on_slider_changed(entry.auto_split_x)


def initSplitImage(self):
    source_dir = Path(self.fileurl)
    destination_dir = Path(self.directoryName)
    destination_dir.mkdir(parents=True, exist_ok=True)

    files = iter_image_files(source_dir)
    total = len(files)
    processed = 0

    self._manual_split_entries = []

    if not files:
        return

    print("__СТАРТ РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ__")

    for file_path in files:
        result = self.parseImage(file_path)
        if result:
            processed += 1
            self.proc.emit(int(processed * 100 / total))

    manual_entries = getattr(self, "_manual_split_entries", [])
    if getattr(self, "isManualSplitAdjust", False) and manual_entries:
        if hasattr(self, "log"):
            self.log.emit("Ожидание ручной корректировки середины")

        wait_event = Event()
        payload = {"entries": manual_entries, "event": wait_event}
        self.manualAdjustmentRequested.emit(payload)
        wait_event.wait()

        if hasattr(self, "log"):
            self.log.emit("Сохранение результатов ручной корректировки")

        self.width_img = 0
        self.height_img = 0

        for entry in manual_entries:
            result = split_with_fixed_position(entry.image, entry.split_x, entry.overlap)
            pages = ((result.left, entry.left_path), (result.right, entry.right_path))
            for page_image, page_path in pages:
                if self.isPxIdentically:
                    if self.width_img and self.height_img:
                        page_image = _fit_page_to_canvas(page_image, self.width_img, self.height_img)
                    else:
                        self.width_img, self.height_img = page_image.shape[1], page_image.shape[0]

                save_with_dpi(page_image, page_path, self.dpi)

        self._manual_split_entries = []


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

    if getattr(self, "isManualSplitAdjust", False):
        if not hasattr(self, "_manual_split_entries"):
            self._manual_split_entries = []

        entry = ManualSplitEntry(
            relative=relative,
            target_dir=target_dir,
            image=image.copy(),
            left_path=left_path,
            right_path=right_path,
            auto_split_x=spread.split_x,
            split_x=spread.split_x,
            overlap=self.width_px,
        )
        self._manual_split_entries.append(entry)
        return str(file_path)

    for page_image, page_path in ((spread.left, left_path), (spread.right, right_path)):
        if self.isPxIdentically:
            if self.width_img and self.height_img:
                page_image = _fit_page_to_canvas(page_image, self.width_img, self.height_img)
            else:
                self.width_img, self.height_img = page_image.shape[1], page_image.shape[0]

        save_with_dpi(page_image, page_path, self.dpi)

    return str(left_path)
