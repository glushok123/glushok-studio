import os
from dataclasses import dataclass, field
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QGridLayout,
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
    base_image: np.ndarray
    left_path: Path
    right_path: Path
    auto_split_x: int
    split_x: int
    overlap: int
    crop_left: int = 0
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0
    rotation_deg: float = 0.0
    split_x_base: int = field(init=False)

    @property
    def width(self) -> int:
        return int(self.base_image.shape[1]) if self.base_image.size else 0

    @property
    def height(self) -> int:
        return int(self.base_image.shape[0]) if self.base_image.size else 0

    def __post_init__(self) -> None:
        self.crop_left = max(0, int(self.crop_left))
        self.crop_top = max(0, int(self.crop_top))
        self.crop_right = self.crop_right or self.width
        self.crop_bottom = self.crop_bottom or self.height
        self.crop_right = max(self.crop_left + 1, int(self.crop_right))
        self.crop_bottom = max(self.crop_top + 1, int(self.crop_bottom))
        self.rotation_deg = float(self.rotation_deg)
        self.split_x = int(np.clip(self.split_x, 0, self.current_width - 1)) if self.current_width else 0
        self.split_x_base = int(self.crop_left + self.split_x)

    @property
    def current_width(self) -> int:
        return max(1, int(self.crop_right - self.crop_left))

    @property
    def current_height(self) -> int:
        return max(1, int(self.crop_bottom - self.crop_top))

    def build_image(self) -> np.ndarray:
        crop_view = self.base_image[
            self.crop_top : self.crop_bottom, self.crop_left : self.crop_right
        ]
        if not crop_view.size:
            return self.base_image.copy()

        crop = crop_view.copy()

        if abs(self.rotation_deg) < 1e-3:
            return crop

        centre = (crop.shape[1] / 2.0, crop.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, self.rotation_deg, 1.0)
        rotated = cv2.warpAffine(
            crop,
            matrix,
            (crop.shape[1], crop.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated


def _numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        return QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


class ManualSplitDialog(QDialog):
    ROTATION_STEP = 0.5

    def __init__(self, entries: List[ManualSplitEntry], parent=None):
        super().__init__(parent)
        self.entries = entries
        self.current_index = 0
        self._pixmap_item = None
        self._split_line = None
        self._current_pixmap = None
        self._grid_lines: list = []
        self._updating = False

        self.setWindowTitle("Ручная корректировка середины")
        self.resize(900, 600)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)

        self.fileLabel = QLabel()
        self.positionLabel = QLabel()
        self.rotationLabel = QLabel()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)

        self.resetButton = QPushButton("Сбросить настройки")
        self.prevButton = QPushButton("Предыдущее")
        self.nextButton = QPushButton("Следующее")
        self.finishButton = QPushButton("Готово")
        self.cancelButton = QPushButton("Отмена")
        self.fullscreenButton = QPushButton("На весь экран")
        self.fullscreenButton.setCheckable(True)
        self.gridButton = QPushButton("Сетка")
        self.gridButton.setCheckable(True)
        self.rotateLeftButton = QPushButton("⟲")
        self.rotateRightButton = QPushButton("⟳")
        self.resetRotationButton = QPushButton("Сброс поворота")

        self.rotateLeftButton.setToolTip("Повернуть на 0.5° влево")
        self.rotateRightButton.setToolTip("Повернуть на 0.5° вправо")
        self.resetRotationButton.setToolTip("Сбросить угол поворота")
        self.gridButton.setToolTip("Показать вспомогательную сетку")
        self.fullscreenButton.setToolTip("Переключить полноэкранный режим окна")

        self.resetButton.clicked.connect(self.reset_current)
        self.prevButton.clicked.connect(self.goto_previous)
        self.nextButton.clicked.connect(self.goto_next)
        self.finishButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)
        self.fullscreenButton.toggled.connect(self.toggle_fullscreen)
        self.gridButton.toggled.connect(self.refresh_scene)
        self.rotateLeftButton.clicked.connect(lambda: self.adjust_rotation(-self.ROTATION_STEP))
        self.rotateRightButton.clicked.connect(lambda: self.adjust_rotation(self.ROTATION_STEP))
        self.resetRotationButton.clicked.connect(self.reset_rotation)

        cropGroup = QGroupBox("Границы")
        cropLayout = QGridLayout()
        cropGroup.setLayout(cropLayout)

        self.leftSpin = QSpinBox()
        self.rightSpin = QSpinBox()
        self.topSpin = QSpinBox()
        self.bottomSpin = QSpinBox()

        for spin in (self.leftSpin, self.rightSpin, self.topSpin, self.bottomSpin):
            spin.setMaximum(99999)
            spin.valueChanged.connect(self.on_crop_changed)

        cropLayout.addWidget(QLabel("Слева"), 0, 0)
        cropLayout.addWidget(self.leftSpin, 0, 1)
        cropLayout.addWidget(QLabel("Справа"), 0, 2)
        cropLayout.addWidget(self.rightSpin, 0, 3)
        cropLayout.addWidget(QLabel("Сверху"), 1, 0)
        cropLayout.addWidget(self.topSpin, 1, 1)
        cropLayout.addWidget(QLabel("Снизу"), 1, 2)
        cropLayout.addWidget(self.bottomSpin, 1, 3)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.prevButton)
        buttonLayout.addWidget(self.nextButton)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.resetButton)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.cancelButton)
        buttonLayout.addWidget(self.finishButton)

        headerLayout = QHBoxLayout()
        headerLayout.addWidget(self.fileLabel)
        headerLayout.addStretch(1)
        headerLayout.addWidget(self.fullscreenButton)

        rotationLayout = QHBoxLayout()
        rotationLayout.addWidget(self.rotateLeftButton)
        rotationLayout.addWidget(self.rotateRightButton)
        rotationLayout.addWidget(self.resetRotationButton)
        rotationLayout.addWidget(self.gridButton)
        rotationLayout.addStretch(1)
        rotationLayout.addWidget(self.rotationLabel)

        layout = QVBoxLayout()
        layout.addLayout(headerLayout)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.positionLabel)
        layout.addWidget(self.slider)
        layout.addLayout(rotationLayout)
        layout.addWidget(cropGroup)
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
        self._updating = True

        max_width = max(1, entry.width - 1)
        max_height = max(1, entry.height - 1)

        self.leftSpin.setMaximum(max_width)
        self.rightSpin.setMaximum(entry.width)
        self.topSpin.setMaximum(max_height)
        self.bottomSpin.setMaximum(entry.height)

        self.leftSpin.setValue(entry.crop_left)
        self.rightSpin.setValue(entry.crop_right)
        self.topSpin.setValue(entry.crop_top)
        self.bottomSpin.setValue(entry.crop_bottom)

        slider_max = max(0, entry.current_width - 1)
        entry.split_x = int(min(entry.split_x, slider_max))
        entry.split_x_base = int(entry.crop_left + entry.split_x)
        self.slider.blockSignals(True)
        self.slider.setMaximum(slider_max)
        self.slider.setValue(min(entry.split_x, slider_max))
        self.slider.blockSignals(False)

        self.slider.setEnabled(entry.current_width > 1)

        self._updating = False

        self.fileLabel.setText(f"{entry.relative.name} ({self.current_index + 1}/{len(self.entries)})")
        self.update_navigation()
        self.refresh_scene()

    def update_navigation(self):
        self.prevButton.setEnabled(self.current_index > 0)
        self.nextButton.setEnabled(self.current_index < len(self.entries) - 1)

    def on_slider_changed(self, value: int):
        entry = self.entries[self.current_index]
        entry.split_x = int(value)
        entry.split_x_base = int(entry.crop_left + entry.split_x)
        self.positionLabel.setText(
            f"Позиция разреза: {entry.split_x}px из {entry.current_width}px"
        )
        if self._split_line is not None:
            self._split_line.setLine(entry.split_x, 0, entry.split_x, entry.current_height)

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
        entry.crop_left = 0
        entry.crop_top = 0
        entry.crop_right = entry.width
        entry.crop_bottom = entry.height
        entry.rotation_deg = 0.0
        entry.split_x = int(np.clip(entry.auto_split_x, 0, entry.current_width - 1))
        entry.split_x_base = entry.crop_left + entry.split_x

        self.display_current_entry()

    def reset_rotation(self):
        entry = self.entries[self.current_index]
        entry.rotation_deg = 0.0
        self.refresh_scene()

    def adjust_rotation(self, delta: float):
        entry = self.entries[self.current_index]
        entry.rotation_deg = float(entry.rotation_deg + delta)
        self.refresh_scene()

    def on_crop_changed(self):
        if self._updating:
            return

        entry = self.entries[self.current_index]

        width = entry.width
        height = entry.height

        left = max(0, min(self.leftSpin.value(), width - 2))
        right = max(left + 1, min(self.rightSpin.value(), width))
        top = max(0, min(self.topSpin.value(), height - 2))
        bottom = max(top + 1, min(self.bottomSpin.value(), height))

        entry.crop_left = left
        entry.crop_right = right
        entry.crop_top = top
        entry.crop_bottom = bottom

        entry.split_x_base = int(np.clip(entry.split_x_base, entry.crop_left, entry.crop_right - 1))
        entry.split_x = int(entry.split_x_base - entry.crop_left)

        self._updating = True
        slider_max = max(0, entry.current_width - 1)
        self.slider.setMaximum(slider_max)
        self.slider.setValue(min(entry.split_x, slider_max))
        self.slider.setEnabled(entry.current_width > 1)
        self.leftSpin.setValue(entry.crop_left)
        self.rightSpin.setValue(entry.crop_right)
        self.topSpin.setValue(entry.crop_top)
        self.bottomSpin.setValue(entry.crop_bottom)
        self._updating = False

        self.refresh_scene()

    def toggle_fullscreen(self, checked: bool):
        if checked:
            self.showFullScreen()
            self.fullscreenButton.setText("Обычный размер")
        else:
            self.showNormal()
            self.fullscreenButton.setText("На весь экран")

    def refresh_scene(self, _checked: bool | None = None):
        if not self.entries:
            return

        entry = self.entries[self.current_index]
        image = entry.build_image()

        self.scene.clear()
        self._grid_lines = []
        self._pixmap_item = None
        self._split_line = None

        pixmap = QPixmap.fromImage(_numpy_to_qimage(image))
        self._current_pixmap = pixmap
        self._pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        pen = QPen(Qt.red)
        pen.setWidth(max(2, pixmap.width() // 400))
        self._split_line = self.scene.addLine(
            entry.split_x, 0, entry.split_x, pixmap.height(), pen
        )

        self._draw_grid(pixmap.width(), pixmap.height())

        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.positionLabel.setText(
            f"Позиция разреза: {entry.split_x}px из {entry.current_width}px"
        )
        self.rotationLabel.setText(f"Поворот: {entry.rotation_deg:.2f}°")

    def _draw_grid(self, width: int, height: int) -> None:
        if not self.gridButton.isChecked():
            return

        pen = QPen(Qt.white)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(max(1, width // 500))
        pen.setColor(Qt.white)

        vertical_step = width / 3.0
        horizontal_step = height / 3.0

        for i in range(1, 3):
            x = i * vertical_step
            line = self.scene.addLine(x, 0, x, height, pen)
            self._grid_lines.append(line)

        for i in range(1, 3):
            y = i * horizontal_step
            line = self.scene.addLine(0, y, width, y, pen)
            self._grid_lines.append(line)

    def closeEvent(self, event):
        if self.fullscreenButton.isChecked():
            self.fullscreenButton.setChecked(False)
        super().closeEvent(event)


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
            final_image = entry.build_image()
            result = split_with_fixed_position(final_image, entry.split_x, entry.overlap)
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
            base_image=image.copy(),
            left_path=left_path,
            right_path=right_path,
            auto_split_x=spread.split_x,
            split_x=spread.split_x,
            overlap=self.width_px,
            crop_left=0,
            crop_top=0,
            crop_right=image.shape[1],
            crop_bottom=image.shape[0],
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
