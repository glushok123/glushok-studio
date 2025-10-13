import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPen, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
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
    split_ratio: float = field(init=False)
    _preview_qimage: QImage | None = field(default=None, init=False, repr=False, compare=False)

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
        if self.current_width > 0:
            self.split_x = int(np.clip(self.split_x, 0, self.current_width - 1))
            self.split_ratio = float(np.clip(self.split_x / self.current_width, 0.0, 1.0))
        else:
            self.split_x = 0
            self.split_ratio = 0.5
        self.split_x_base = int(self.crop_left + self.split_x)

    def preview_qimage(self) -> QImage:
        if self._preview_qimage is None:
            self._preview_qimage = _numpy_to_qimage(self.base_image)
        return self._preview_qimage

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
        crop_view = self.base_image[
            self.crop_top : self.crop_bottom, self.crop_left : self.crop_right
        ]
        if not crop_view.size:
            return self.base_image.copy()

        crop = crop_view.copy()

        if abs(self.rotation_deg) < 1e-3:
            return crop

        height, width = crop.shape[:2]
        centre = (width / 2.0, height / 2.0)

        matrix = cv2.getRotationMatrix2D(centre, self.rotation_deg, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])

        new_width = int(np.ceil(width * cos + height * sin))
        new_height = int(np.ceil(width * sin + height * cos))

        matrix[0, 2] += new_width / 2.0 - centre[0]
        matrix[1, 2] += new_height / 2.0 - centre[1]

        rotated = cv2.warpAffine(
            crop,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        mask = np.ones((height, width), dtype=np.uint8) * 255
        rotated_mask = cv2.warpAffine(
            mask,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        coords = cv2.findNonZero(rotated_mask)
        if coords is None:
            return rotated

        x, y, w_box, h_box = cv2.boundingRect(coords)
        trimmed = rotated[y : y + h_box, x : x + w_box]
        return trimmed if trimmed.size else rotated

    def final_split_position(self, final_width: int) -> int:
        if final_width <= 1:
            return 0

        ratio = float(np.clip(self.split_ratio, 0.0, 1.0))
        position = int(round(ratio * (final_width - 1)))
        return int(np.clip(position, 0, final_width - 1))


def _numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        return QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


class CropHandle(QObject, QGraphicsRectItem):
    moved = pyqtSignal(float)

    def __init__(self, orientation: str, size: float = 18.0, parent: QObject | None = None):
        QObject.__init__(self, parent)
        QGraphicsRectItem.__init__(self)
        self.orientation = orientation
        self.size = size
        self._minimum = -float("inf")
        self._maximum = float("inf")
        self._fixed = 0.0

        half = size / 2.0
        self.setRect(-half, -half, size, size)

        pen = QPen(QColor(255, 255, 255, 220))
        pen.setWidth(1)
        self.setPen(pen)
        self.setBrush(QColor(122, 215, 255, 180))
        self.setZValue(10)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setCursor(
            Qt.SizeHorCursor if orientation in {"left", "right"} else Qt.SizeVerCursor
        )

    def set_limits(self, minimum: float, maximum: float, fixed_coord: float) -> None:
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._fixed = float(fixed_coord)

    def itemChange(self, change: "QGraphicsItem.GraphicsItemChange", value):
        if change == QGraphicsItem.ItemPositionChange:
            point: QPointF = value
            if self.orientation in {"left", "right"}:
                x = max(self._minimum, min(self._maximum, point.x()))
                return QPointF(x, self._fixed)
            else:
                y = max(self._minimum, min(self._maximum, point.y()))
                return QPointF(self._fixed, y)

        if change == QGraphicsItem.ItemPositionHasChanged:
            coord = self.pos().x() if self.orientation in {"left", "right"} else self.pos().y()
            self.moved.emit(float(coord))

        return super().itemChange(change, value)


class ManualSplitDialog(QDialog):
    ROTATION_STEP = 0.5

    def __init__(self, entries: List[ManualSplitEntry], parent=None):
        super().__init__(parent)
        self.entries = entries
        self.current_index = 0

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._split_line: QGraphicsLineItem | None = None
        self._crop_rect_item: QGraphicsRectItem | None = None
        self._handles: dict[str, CropHandle] = {}
        self._grid_lines: list[QGraphicsLineItem] = []
        self._loaded_entry: ManualSplitEntry | None = None
        self._updating = False

        self.setWindowTitle("Ручная корректировка середины")
        self.resize(900, 600)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setAlignment(Qt.AlignCenter)

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
            spin.setMaximum(999999)
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
            for widget in (
                self.slider,
                self.prevButton,
                self.nextButton,
                self.resetButton,
                self.finishButton,
            ):
                widget.setEnabled(False)

    def display_current_entry(self):
        entry = self.entries[self.current_index]
        self.fileLabel.setText(
            f"{entry.relative.name} ({self.current_index + 1}/{len(self.entries)})"
        )
        self.update_navigation()
        self._sync_controls(entry)
        self.refresh_scene()

    def update_navigation(self):
        self.prevButton.setEnabled(self.current_index > 0)
        self.nextButton.setEnabled(self.current_index < len(self.entries) - 1)

    def on_slider_changed(self, value: int):
        if self._updating or not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.set_split_x(int(value))
        self.positionLabel.setText(
            f"Позиция разреза: {entry.split_x}px из {entry.current_width}px"
        )
        if self._split_line is not None:
            self._split_line.setLine(
                entry.split_x_base, entry.crop_top, entry.split_x_base, entry.crop_bottom
            )

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
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.crop_left = 0
        entry.crop_top = 0
        entry.crop_right = entry.width
        entry.crop_bottom = entry.height
        entry.rotation_deg = 0.0
        auto_split = int(np.clip(entry.auto_split_x, 0, entry.width - 1)) if entry.width > 1 else 0
        entry.set_split_x(auto_split)
        self._sync_controls(entry)
        self.refresh_scene()

    def reset_rotation(self):
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.rotation_deg = 0.0
        self.refresh_scene()

    def adjust_rotation(self, delta: float):
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.rotation_deg = float(entry.rotation_deg + delta)
        self.refresh_scene()

    def on_crop_changed(self):
        if self._updating or not self.entries:
            return
        entry = self.entries[self.current_index]
        self._apply_crop_change(
            entry,
            left=self.leftSpin.value(),
            right=self.rightSpin.value(),
            top=self.topSpin.value(),
            bottom=self.bottomSpin.value(),
        )

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
        self._load_entry(entry)
        self._update_pixmap(entry)
        self._update_overlay(entry)
        self._update_grid(entry)
        self._update_scene_rect()

        self.view.resetTransform()
        rect = self.scene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.view.fitInView(rect, Qt.KeepAspectRatio)

        self.positionLabel.setText(
            f"Позиция разреза: {entry.split_x}px из {entry.current_width}px"
        )
        self.rotationLabel.setText(f"Поворот: {entry.rotation_deg:.2f}°")

    def _load_entry(self, entry: ManualSplitEntry) -> None:
        if self._pixmap_item is None:
            pixmap = QPixmap.fromImage(entry.preview_qimage())
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            self.scene.addItem(self._pixmap_item)
        elif self._loaded_entry is not entry:
            self._pixmap_item.setPixmap(QPixmap.fromImage(entry.preview_qimage()))

        if self._crop_rect_item is None:
            pen = QPen(QColor(239, 83, 80))
            pen.setWidth(2)
            self._crop_rect_item = self.scene.addRect(QRectF(), pen)
            self._crop_rect_item.setZValue(2)

        if self._split_line is None:
            pen = QPen(Qt.red)
            pen.setWidth(2)
            self._split_line = self.scene.addLine(0, 0, 0, 0, pen)
            self._split_line.setZValue(3)

        if not self._handles:
            for side in ("left", "right", "top", "bottom"):
                handle = CropHandle(side)
                handle.moved.connect(lambda coord, s=side: self._handle_moved(s, coord))
                self.scene.addItem(handle)
                self._handles[side] = handle

        self._loaded_entry = entry

    def _update_pixmap(self, entry: ManualSplitEntry) -> None:
        if not self._pixmap_item:
            return

        if self._loaded_entry is not entry:
            self._pixmap_item.setPixmap(QPixmap.fromImage(entry.preview_qimage()))
            self._loaded_entry = entry

        centre_x, centre_y = entry.crop_centre
        self._pixmap_item.setTransformOriginPoint(QPointF(centre_x, centre_y))
        self._pixmap_item.setRotation(entry.rotation_deg)

    def _update_overlay(self, entry: ManualSplitEntry) -> None:
        if not self._crop_rect_item or not self._split_line:
            return

        rect = QRectF(entry.crop_left, entry.crop_top, entry.current_width, entry.current_height)
        rect_pen = self._crop_rect_item.pen()
        rect_pen.setWidth(max(2, int(round(rect.width() / 200)) + 1))
        self._crop_rect_item.setPen(rect_pen)
        self._crop_rect_item.setRect(rect)

        split_pen = self._split_line.pen()
        split_pen.setWidth(max(2, entry.current_width // 200 + 1))
        self._split_line.setPen(split_pen)
        self._split_line.setLine(
            entry.split_x_base, entry.crop_top, entry.split_x_base, entry.crop_bottom
        )

        self._update_handles(entry, rect)

    def _update_handles(self, entry: ManualSplitEntry, rect: QRectF) -> None:
        if not self._handles:
            return

        min_width = 2
        min_height = 2
        centre_x = rect.center().x()
        centre_y = rect.center().y()

        left_handle = self._handles["left"]
        left_handle.blockSignals(True)
        left_handle.set_limits(0, max(0.0, entry.crop_right - min_width), centre_y)
        left_handle.setPos(entry.crop_left, centre_y)
        left_handle.blockSignals(False)
        left_handle.setVisible(entry.current_width > min_width)

        right_handle = self._handles["right"]
        right_handle.blockSignals(True)
        right_handle.set_limits(entry.crop_left + min_width, entry.width, centre_y)
        right_handle.setPos(entry.crop_right, centre_y)
        right_handle.blockSignals(False)
        right_handle.setVisible(entry.current_width > min_width)

        top_handle = self._handles["top"]
        top_handle.blockSignals(True)
        top_handle.set_limits(0, max(0.0, entry.crop_bottom - min_height), centre_x)
        top_handle.setPos(centre_x, entry.crop_top)
        top_handle.blockSignals(False)
        top_handle.setVisible(entry.current_height > min_height)

        bottom_handle = self._handles["bottom"]
        bottom_handle.blockSignals(True)
        bottom_handle.set_limits(entry.crop_top + min_height, entry.height, centre_x)
        bottom_handle.setPos(centre_x, entry.crop_bottom)
        bottom_handle.blockSignals(False)
        bottom_handle.setVisible(entry.current_height > min_height)

    def _handle_moved(self, side: str, coordinate: float) -> None:
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        rounded = int(round(coordinate))
        if side == "left":
            self._apply_crop_change(entry, left=rounded)
        elif side == "right":
            self._apply_crop_change(entry, right=rounded)
        elif side == "top":
            self._apply_crop_change(entry, top=rounded)
        elif side == "bottom":
            self._apply_crop_change(entry, bottom=rounded)

    def _apply_crop_change(
        self,
        entry: ManualSplitEntry,
        *,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
    ) -> None:
        width = entry.width
        height = entry.height

        left_val = entry.crop_left if left is None else int(left)
        right_val = entry.crop_right if right is None else int(right)
        top_val = entry.crop_top if top is None else int(top)
        bottom_val = entry.crop_bottom if bottom is None else int(bottom)

        left_val = max(0, min(left_val, width - 1))
        right_val = max(1, min(right_val, width))
        top_val = max(0, min(top_val, height - 1))
        bottom_val = max(1, min(bottom_val, height))

        min_width = 2
        min_height = 2

        if right_val - left_val < min_width:
            if left is None:
                left_val = max(0, right_val - min_width)
            elif right is None:
                right_val = min(width, left_val + min_width)
            else:
                right_val = min(width, left_val + min_width)

        if bottom_val - top_val < min_height:
            if top is None:
                top_val = max(0, bottom_val - min_height)
            elif bottom is None:
                bottom_val = min(height, top_val + min_height)
            else:
                bottom_val = min(height, top_val + min_height)

        entry.crop_left = left_val
        entry.crop_right = right_val
        entry.crop_top = top_val
        entry.crop_bottom = bottom_val

        entry.update_split_from_ratio()
        self._sync_controls(entry)
        self.refresh_scene()

    def _sync_controls(self, entry: ManualSplitEntry) -> None:
        self._updating = True

        width = entry.width
        height = entry.height

        self.leftSpin.setMaximum(max(0, width - 1))
        self.rightSpin.setMaximum(width)
        self.topSpin.setMaximum(max(0, height - 1))
        self.bottomSpin.setMaximum(height)

        self.leftSpin.setValue(entry.crop_left)
        self.rightSpin.setValue(entry.crop_right)
        self.topSpin.setValue(entry.crop_top)
        self.bottomSpin.setValue(entry.crop_bottom)

        slider_max = max(0, entry.current_width - 1)
        self.slider.setMaximum(slider_max)
        self.slider.setEnabled(slider_max > 0)
        current_value = min(entry.split_x, slider_max) if slider_max >= 0 else 0
        self.slider.setValue(current_value)

        self._updating = False

        self.positionLabel.setText(
            f"Позиция разреза: {entry.split_x}px из {entry.current_width}px"
        )
        self.rotationLabel.setText(f"Поворот: {entry.rotation_deg:.2f}°")

    def _update_grid(self, entry: ManualSplitEntry) -> None:
        for line in self._grid_lines:
            self.scene.removeItem(line)
        self._grid_lines.clear()

        if not self.gridButton.isChecked():
            return

        width = entry.current_width
        height = entry.current_height
        if width <= 1 or height <= 1:
            return

        pen = QPen(QColor(255, 255, 255, 180))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(max(1, width // 200 + 1))

        left = entry.crop_left
        right = entry.crop_right
        top = entry.crop_top
        bottom = entry.crop_bottom

        for i in range(1, 3):
            x = left + width * i / 3.0
            line = self.scene.addLine(x, top, x, bottom, pen)
            self._grid_lines.append(line)

        for i in range(1, 3):
            y = top + height * i / 3.0
            line = self.scene.addLine(left, y, right, y, pen)
            self._grid_lines.append(line)

    def _update_scene_rect(self) -> None:
        rect = self.scene.itemsBoundingRect()
        if rect.isNull():
            return
        padding = 40.0
        rect = rect.adjusted(-padding, -padding, padding, padding)
        self.scene.setSceneRect(rect)

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
            final_image = entry.build_processed_image()
            final_width = final_image.shape[1] if final_image.ndim >= 2 else 0
            split_position = entry.final_split_position(final_width)
            result = split_with_fixed_position(final_image, split_position, entry.overlap)
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
