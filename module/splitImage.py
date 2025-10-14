import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from threading import Event

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPen, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsObject,
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

        # ``QGraphicsItem`` uses a screen coordinate system where positive
        # angles rotate the pixmap clockwise.  OpenCV, on the other hand,
        # interprets positive angles as counter-clockwise rotations.  Without
        # compensating for the sign difference the preview inside the manual
        # adjustment dialog does not match the saved image – pages rotated to
        # the right end up rotated to the left after export.  Negating the
        # angle here keeps the on-screen transform and the saved result in
        # sync.
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


def _numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        return QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


class CropHandle(QGraphicsObject):
    moved = pyqtSignal(float)

    def __init__(self, orientation: str, size: float = 18.0, parent: QObject | None = None):
        super().__init__(parent)
        self.orientation = orientation
        self.size = size
        self._minimum = -float("inf")
        self._maximum = float("inf")
        self._fixed = 0.0
        self._rect = QRectF(-size / 2.0, -size / 2.0, size, size)
        self._pen = QPen(QColor(255, 255, 255, 220))
        self._pen.setWidth(1)
        self._brush = QColor(122, 215, 255, 180)

        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setCursor(
            Qt.SizeHorCursor if orientation in {"left", "right"} else Qt.SizeVerCursor
        )

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        painter.setPen(self._pen)
        painter.setBrush(self._brush)
        painter.drawRect(self._rect)

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

    def __init__(
        self,
        entries: List[ManualSplitEntry],
        parent=None,
        *,
        identical_resolution: bool = False,
        target_page_width: int | None = None,
        target_page_height: int | None = None,
    ):
        super().__init__(parent)
        self.entries = entries
        self.current_index = 0

        self._identical_resolution = bool(identical_resolution)
        self._target_page_width = int(target_page_width or 0)
        self._target_page_height = int(target_page_height or 0)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._split_line: QGraphicsLineItem | None = None
        self._crop_rect_item: QGraphicsRectItem | None = None
        self._handles: dict[str, CropHandle] = {}
        self._grid_lines: list[QGraphicsLineItem] = []
        self._loaded_entry: ManualSplitEntry | None = None
        self._updating = False
        self._last_loaded_index: int | None = None
        self._handling_crop_change = False

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

        self.splitToggle = QCheckBox("Разделить изображение")
        self.splitToggle.setChecked(True)
        self.splitToggle.toggled.connect(self.on_split_toggled)

        self.alignResolutionButton = QPushButton("Выставить по разрешению")
        self.alignResolutionButton.setToolTip(
            "Подогнать рамку под сохранённое разрешение страниц"
        )
        self.alignResolutionButton.clicked.connect(self.align_to_resolution)
        self.alignResolutionButton.setVisible(self._identical_resolution)
        self.alignResolutionButton.setEnabled(False)

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

        splitControlLayout = QHBoxLayout()
        splitControlLayout.addWidget(self.splitToggle)
        splitControlLayout.addWidget(self.slider, 1)
        splitControlLayout.addWidget(self.alignResolutionButton)

        layout = QVBoxLayout()
        layout.addLayout(headerLayout)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.positionLabel)
        layout.addLayout(splitControlLayout)
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
        if (
            self._last_loaded_index is not None
            and 0 <= self._last_loaded_index < len(self.entries)
            and self._last_loaded_index != self.current_index
        ):
            try:
                self.entries[self._last_loaded_index].release_image()
            except Exception:
                pass

        entry = self.entries[self.current_index]
        self._last_loaded_index = self.current_index
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
        if entry.split_disabled:
            return
        entry.set_split_x(int(value))
        if self._split_line is not None:
            self._split_line.setLine(
                entry.split_x_base, entry.crop_top, entry.split_x_base, entry.crop_bottom
            )
        self._update_split_status(entry)

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
        entry.split_disabled = False
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
        if self._pixmap_item is not None:
            self.view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        else:
            rect = self.scene.sceneRect()
            if rect.width() > 0 and rect.height() > 0:
                self.view.fitInView(rect, Qt.KeepAspectRatio)

        self.rotationLabel.setText(f"Поворот: {entry.rotation_deg:.2f}°")
        self._update_split_status(entry)

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

        if entry.split_disabled:
            self._split_line.setVisible(False)
        else:
            split_pen = self._split_line.pen()
            split_pen.setWidth(max(2, entry.current_width // 200 + 1))
            self._split_line.setPen(split_pen)
            self._split_line.setLine(
                entry.split_x_base, entry.crop_top, entry.split_x_base, entry.crop_bottom
            )
            self._split_line.setVisible(True)

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
        if not self.entries or self._handling_crop_change:
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
        if self._handling_crop_change:
            return

        self._handling_crop_change = True
        try:
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
        finally:
            self._handling_crop_change = False

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

        self.rotationLabel.setText(f"Поворот: {entry.rotation_deg:.2f}°")
        self._update_split_status(entry)

    def on_split_toggled(self, checked: bool) -> None:
        if self._updating or not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.split_disabled = not bool(checked)
        self._update_split_status(entry)
        self.refresh_scene()

    def align_to_resolution(self) -> None:
        if not self.entries or not self._identical_resolution:
            return
        entry = self.entries[self.current_index]
        if entry.split_disabled:
            return

        target_width = int(self._target_page_width)
        target_height = int(self._target_page_height)
        if target_width <= 0 or target_height <= 0:
            final_image = entry.build_processed_image()
            final_width = final_image.shape[1] if final_image.ndim >= 2 else 0
            split_position = entry.final_split_position(final_width)
            split_result = split_with_fixed_position(final_image, split_position, entry.overlap)
            if split_result.left.size:
                self.set_target_resolution(
                    split_result.left.shape[1],
                    split_result.left.shape[0],
                )
            return

        overlap = max(0, int(entry.overlap))
        half_width = max(1, target_width - overlap)
        base = entry.split_x_base
        new_width = max(2, half_width * 2)

        if entry.width < new_width:
            crop_left = 0
            crop_right = entry.width
        else:
            crop_left = base - half_width
            crop_right = crop_left + new_width
            if crop_left < 0:
                crop_left = 0
                crop_right = new_width
            if crop_right > entry.width:
                crop_right = entry.width
                crop_left = entry.width - new_width

        crop_left = int(round(crop_left))
        crop_right = int(round(crop_right))
        crop_left = max(0, min(crop_left, entry.width - 1))
        crop_right = max(crop_left + 1, min(crop_right, entry.width))

        desired_height = min(target_height, entry.height)
        current_height = entry.current_height
        centre_y = entry.crop_top + current_height / 2.0 if current_height else entry.crop_top
        half_height = desired_height / 2.0
        crop_top = int(round(centre_y - half_height))
        crop_bottom = crop_top + desired_height
        if crop_top < 0:
            crop_top = 0
            crop_bottom = desired_height
        if crop_bottom > entry.height:
            crop_bottom = entry.height
            crop_top = entry.height - desired_height
        crop_top = max(0, min(int(round(crop_top)), entry.height - 1))
        crop_bottom = max(crop_top + 1, min(int(round(crop_bottom)), entry.height))

        entry.crop_left = crop_left
        entry.crop_right = crop_right
        entry.crop_top = crop_top
        entry.crop_bottom = crop_bottom
        entry.set_split_base(base)

        self._sync_controls(entry)
        self.refresh_scene()

    def _update_split_status(self, entry: ManualSplitEntry) -> None:
        slider_has_range = self.slider.maximum() > 0
        slider_enabled = slider_has_range and not entry.split_disabled
        self.slider.setEnabled(slider_enabled)

        text = "Разделение отключено" if entry.split_disabled else (
            f"Позиция разреза: {entry.split_x}px из {entry.current_width}px"
        )
        self.positionLabel.setText(text)

        self.splitToggle.blockSignals(True)
        self.splitToggle.setChecked(not entry.split_disabled)
        self.splitToggle.blockSignals(False)

        can_align = (
            self._identical_resolution
            and not entry.split_disabled
            and self._target_page_width > 0
            and self._target_page_height > 0
        )
        self.alignResolutionButton.setEnabled(can_align)
        self.alignResolutionButton.setVisible(self._identical_resolution)

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
        if self._pixmap_item is not None:
            rect = self._pixmap_item.sceneBoundingRect()
        else:
            rect = self.scene.itemsBoundingRect()
        if rect.isNull():
            return
        padding = 40.0
        rect = rect.adjusted(-padding, -padding, padding, padding)
        self.scene.setSceneRect(rect)

    @property
    def target_page_width(self) -> int:
        return max(0, int(self._target_page_width))

    @property
    def target_page_height(self) -> int:
        return max(0, int(self._target_page_height))

    def set_target_resolution(self, width: int | None, height: int | None) -> None:
        self._target_page_width = int(width or 0)
        self._target_page_height = int(height or 0)
        if self.entries:
            self._update_split_status(self.entries[self.current_index])

    def closeEvent(self, event):
        if self.fullscreenButton.isChecked():
            self.fullscreenButton.setChecked(False)
        for entry in self.entries:
            try:
                entry.release_image()
            except Exception:
                pass
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
        print(f"[INFO] Найдено {len(manual_entries)} файлов для ручной корректировки")

        wait_event = Event()
        result_holder = {"accepted": False, "width_img": self.width_img, "height_img": self.height_img}
        payload = {
            "entries": manual_entries,
            "wait_event": wait_event,
            "result": result_holder,
            "thread": self,
        }
        self.manualAdjustmentRequested.emit(payload)
        wait_event.wait()
        accepted = bool(result_holder.get("accepted"))
        if "width_img" in result_holder:
            try:
                self.width_img = int(result_holder["width_img"] or 0)
            except Exception:
                pass
        if "height_img" in result_holder:
            try:
                self.height_img = int(result_holder["height_img"] or 0)
            except Exception:
                pass
        print("[INFO] Слот ручной корректировки завершил работу")

        if hasattr(self, "log"):
            status = "принята" if accepted else "отменена"
            self.log.emit(f"Ручная корректировка {status}, продолжаем обработку")
            self.log.emit("Сохранение результатов ручной корректировки")
        print(f"[INFO] Ручная корректировка { 'принята' if accepted else 'отменена' }")

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

    original_root = getattr(self, "original_fileurl", self.fileurl)
    original_candidate = Path(original_root) / relative
    if original_candidate.exists():
        original_path = original_candidate
    else:
        original_path = file_path

    if original_path == file_path:
        original_image = image
    else:
        try:
            original_image = load_image(original_path)
        except ValueError:
            original_image = image
            original_path = file_path

    manual_mode = bool(getattr(self, "isManualSplitAdjust", False))

    original_height, original_width = original_image.shape[:2]
    manual_crop_left = 0
    manual_crop_top = 0
    manual_crop_right = original_width
    manual_crop_bottom = original_height

    analysis_image = image

    def _extract_manual_view(src: np.ndarray) -> np.ndarray:
        if src is None or not src.size:
            return src

        left = max(0, min(manual_crop_left, src.shape[1] - 1))
        right = max(left + 1, min(manual_crop_right, src.shape[1]))
        top = max(0, min(manual_crop_top, src.shape[0] - 1))
        bottom = max(top + 1, min(manual_crop_bottom, src.shape[0]))

        view = src[top:bottom, left:right]
        return view if view.size else src

    if getattr(self, "isRemoveBorder", False):
        pad = self.border_px if getattr(self, "isAddBorder", False) else 0
        cropped, bounds = crop_to_content(original_image, pad_x=pad, pad_y=pad)
        if cropped is not None and bounds is not None:
            manual_crop_left = int(bounds.left)
            manual_crop_top = int(bounds.top)
            manual_crop_right = int(bounds.right)
            manual_crop_bottom = int(bounds.bottom)
            if manual_mode:
                analysis_image = _extract_manual_view(original_image)
            else:
                image = cropped
                analysis_image = image
        else:
            analysis_image = original_image if manual_mode else image
    else:
        analysis_image = original_image if manual_mode else image

    if analysis_image is None or not analysis_image.size:
        analysis_image = original_image if manual_mode else image

    if manual_mode:
        height, width = analysis_image.shape[:2]
    else:
        height, width = image.shape[:2]

    if height >= width:
        save_path = target_dir / relative.name
        if manual_mode:
            manual_view = _extract_manual_view(original_image)
            save_source = manual_view if manual_view is not None and manual_view.size else original_image
        else:
            save_source = image
        save_with_dpi(save_source, save_path, self.dpi)
        return str(save_path)

    try:
        spread_source = analysis_image if manual_mode else image
        spread = split_spread(spread_source, self.width_px, self.pxMediumVal)
    except Exception as exc:
        print(f"[WARN] Не удалось разделить {file_path}: {exc}")
        save_path = target_dir / relative.name
        if manual_mode:
            manual_view = _extract_manual_view(original_image)
            save_source = manual_view if manual_view is not None and manual_view.size else original_image
        else:
            save_source = analysis_image if analysis_image is not None and analysis_image.size else image
        save_with_dpi(save_source, save_path, self.dpi)
        return str(save_path)

    number = relative.stem
    left_name = f"{number}_1.jpg"
    right_name = f"{number}_2.jpg"

    left_path = target_dir / left_name
    right_path = target_dir / right_name

    if getattr(self, "isManualSplitAdjust", False):
        if not hasattr(self, "_manual_split_entries"):
            self._manual_split_entries = []

        entry_data = {
            "source_path": str(original_path),
            "relative": str(relative),
            "target_dir": str(target_dir),
            "left_path": str(left_path),
            "right_path": str(right_path),
            "auto_split_x": int(spread.split_x),
            "split_x": int(spread.split_x),
            "overlap": int(self.width_px),
            "crop_left": manual_crop_left,
            "crop_top": manual_crop_top,
            "crop_right": manual_crop_right,
            "crop_bottom": manual_crop_bottom,
            "rotation_deg": 0.0,
            "image_width": original_width,
            "image_height": original_height,
            "split_disabled": False,
        }
        self._manual_split_entries.append(entry_data)
        return str(file_path)

    for page_image, page_path in ((spread.left, left_path), (spread.right, right_path)):
        if self.isPxIdentically:
            if self.width_img and self.height_img:
                page_image = _fit_page_to_canvas(page_image, self.width_img, self.height_img)
            else:
                self.width_img, self.height_img = page_image.shape[1], page_image.shape[0]

        save_with_dpi(page_image, page_path, self.dpi)

    return str(left_path)
