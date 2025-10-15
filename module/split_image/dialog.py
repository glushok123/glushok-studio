"""Qt dialog for manual fine-tuning of page splits."""
from __future__ import annotations

from typing import List

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from .entries import (
    ManualSplitEntry,
    _entry_page_dimensions,
    collect_resolution_metrics,
    enforce_entry_targets,
)


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

    def set_limits(self, minimum: float, maximum: float, fixed: float) -> None:
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._fixed = float(fixed)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.ItemPositionChange:
            pos: QPointF = value
            if self.orientation in {"left", "right"}:
                new_x = np.clip(pos.x(), self._minimum, self._maximum)
                return QPointF(new_x, self._fixed)
            if self.orientation in {"top", "bottom"}:
                new_y = np.clip(pos.y(), self._minimum, self._maximum)
                return QPointF(self._fixed, new_y)
        if change == QGraphicsItem.ItemPositionHasChanged:
            if self.orientation in {"left", "right"}:
                self.moved.emit(self.pos().x())
            else:
                self.moved.emit(self.pos().y())
        return super().itemChange(change, value)


class ManualSplitDialog(QDialog):
    ROTATION_STEP = 0.5

    def __init__(
        self,
        entries: List[ManualSplitEntry],
        *,
        identical_resolution: bool = False,
        target_page_width: int = 0,
        target_page_height: int = 0,
        parent=None,
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
        self._resolution_metrics: dict[str, int | str | None] = (
            collect_resolution_metrics(self.entries)
        )

        self.setWindowTitle("Ручная корректировка середины")
        self.resize(900, 600)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._content_group = QGraphicsItemGroup()
        # PyQt5 exposes ``setHandlesChildEvents`` on ``QGraphicsItem`` but not
        # directly on ``QGraphicsItemGroup``.  Calling the base-class helper
        # keeps the child crop handles responsive while still letting us group
        # the overlay items together.
        QGraphicsItem.setHandlesChildEvents(self._content_group, False)
        self._content_group.setFlag(QGraphicsItem.ItemHasNoContents, True)
        self.scene.addItem(self._content_group)

        self._zoom = 1.0
        self._manual_zoom = False
        self._fit_pending = True
        self._ZOOM_MIN = 1
        self._ZOOM_MAX = 1000
        self._ZOOM_STEP = 10

        self.fileLabel = QLabel()
        self.positionLabel = QLabel()
        self.rotationLabel = QLabel()

        self.heightInfoLabel = QLabel()
        self.heightInfoLabel.setWordWrap(True)
        self.widthInfoLabel = QLabel()
        self.widthInfoLabel.setWordWrap(True)
        self.resolutionWarningLabel = QLabel()
        self.resolutionWarningLabel.setWordWrap(True)
        self.resolutionWarningLabel.setStyleSheet("color: #ef5350;")
        self.heightInfoLabel.setVisible(self._identical_resolution)
        self.widthInfoLabel.setVisible(self._identical_resolution)
        self.resolutionWarningLabel.setVisible(False)

        self.gotoHeightButton = QPushButton("Перейти к макс. высоте")
        self.gotoWidthButton = QPushButton("Перейти к макс. ширине")
        self.gotoHeightButton.clicked.connect(lambda: self.goto_metric_entry("height"))
        self.gotoWidthButton.clicked.connect(lambda: self.goto_metric_entry("width"))
        for button in (self.gotoHeightButton, self.gotoWidthButton):
            button.setVisible(self._identical_resolution)
            button.setEnabled(False)

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
        self.gotoFirstButton = QPushButton("В начало")
        self.prevButton = QPushButton("Предыдущее")
        self.nextButton = QPushButton("Следующее")
        self.gotoLastButton = QPushButton("В конец")
        self.finishButton = QPushButton("Готово")
        self.cancelButton = QPushButton("Отмена")
        self.fullscreenButton = QPushButton("На весь экран")
        self.fullscreenButton.setCheckable(True)
        self.gridButton = QPushButton("Сетка")
        self.gridButton.setCheckable(True)
        self.rotateLeftButton = QPushButton("⟲")
        self.rotateRightButton = QPushButton("⟳")
        self.resetRotationButton = QPushButton("Сбросить угол")

        self.rotateLeftButton.setToolTip("Повернуть на 0.5° влево")
        self.rotateRightButton.setToolTip("Повернуть на 0.5° вправо")
        self.resetRotationButton.setToolTip("Сбросить угол поворота")
        self.gridButton.setToolTip("Показать вспомогательную сетку")
        self.fullscreenButton.setToolTip("Переключить полноэкранный режим окна")

        self.gotoIndexSpin = QSpinBox()
        self.gotoIndexSpin.setMinimum(1)
        self.gotoIndexSpin.setMaximum(max(1, len(entries)))
        self.gotoIndexSpin.setKeyboardTracking(False)
        self.gotoIndexSpin.setFixedWidth(80)
        self.gotoIndexButton = QPushButton("Перейти")

        self.zoomLabel = QLabel("Масштаб: 100%")
        self.zoomSlider = QSlider(Qt.Horizontal)
        self.zoomSlider.setRange(self._ZOOM_MIN, self._ZOOM_MAX)
        self.zoomSlider.setValue(100)
        self.zoomSlider.valueChanged.connect(self.on_zoom_slider_changed)
        self.zoomOutButton = QPushButton("−")
        self.zoomOutButton.setToolTip("Уменьшить масштаб")
        self.zoomOutButton.clicked.connect(lambda: self.step_zoom(-self._ZOOM_STEP))
        self.zoomInButton = QPushButton("+")
        self.zoomInButton.setToolTip("Увеличить масштаб")
        self.zoomInButton.clicked.connect(lambda: self.step_zoom(self._ZOOM_STEP))
        self.zoomFitButton = QPushButton("По размеру")
        self.zoomFitButton.setToolTip("Подогнать изображение под размер окна")
        self.zoomFitButton.clicked.connect(self.reset_zoom_to_fit)

        self.resetButton.clicked.connect(self.reset_current)
        self.gotoFirstButton.clicked.connect(self.goto_first)
        self.prevButton.clicked.connect(self.goto_previous)
        self.nextButton.clicked.connect(self.goto_next)
        self.gotoLastButton.clicked.connect(self.goto_last)
        self.gotoIndexButton.clicked.connect(self.goto_index)
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
        buttonLayout.addWidget(self.gotoFirstButton)
        buttonLayout.addWidget(self.prevButton)
        buttonLayout.addWidget(self.gotoIndexSpin)
        buttonLayout.addWidget(self.gotoIndexButton)
        buttonLayout.addWidget(self.nextButton)
        buttonLayout.addWidget(self.gotoLastButton)
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

        zoomControlsLayout = QVBoxLayout()
        zoomButtonsLayout = QHBoxLayout()
        zoomButtonsLayout.addWidget(self.zoomOutButton)
        zoomButtonsLayout.addWidget(self.zoomInButton)
        zoomButtonsLayout.addWidget(self.zoomFitButton)
        zoomControlsLayout.addLayout(zoomButtonsLayout)
        zoomControlsLayout.addWidget(self.zoomSlider)
        zoomControlsLayout.addWidget(self.zoomLabel)

        sideControlsLayout = QVBoxLayout()
        sideControlsLayout.addLayout(zoomControlsLayout)
        sideControlsLayout.addWidget(cropGroup)
        sideControlsLayout.addStretch(1)

        contentLayout = QHBoxLayout()
        contentLayout.addLayout(sideControlsLayout)
        contentLayout.addWidget(self.view, stretch=1)
        contentLayout.setAlignment(sideControlsLayout, Qt.AlignTop)

        layout.addLayout(contentLayout)
        layout.addWidget(self.positionLabel)
        layout.addWidget(self.heightInfoLabel)
        layout.addWidget(self.widthInfoLabel)
        metricsButtonLayout = QHBoxLayout()
        metricsButtonLayout.addWidget(self.gotoHeightButton)
        metricsButtonLayout.addWidget(self.gotoWidthButton)
        metricsButtonLayout.addStretch(1)
        layout.addLayout(metricsButtonLayout)
        layout.addWidget(self.resolutionWarningLabel)
        layout.addLayout(splitControlLayout)
        layout.addLayout(rotationLayout)
        layout.addLayout(buttonLayout)

        self.setLayout(layout)

        if self.entries:
            self.display_current_entry()
        else:
            for widget in (
                self.slider,
                self.gotoFirstButton,
                self.prevButton,
                self.nextButton,
                self.gotoLastButton,
                self.gotoIndexSpin,
                self.gotoIndexButton,
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
        self._update_resolution_metrics()
        self._manual_zoom = False
        self._fit_pending = True
        self.refresh_scene()

    def update_navigation(self):
        total = len(self.entries)
        has_entries = total > 0

        self.prevButton.setEnabled(self.current_index > 0)
        self.nextButton.setEnabled(self.current_index < total - 1)
        self.gotoFirstButton.setEnabled(has_entries and self.current_index > 0)
        self.gotoLastButton.setEnabled(has_entries and self.current_index < total - 1)

        self.gotoIndexSpin.setEnabled(has_entries)
        self.gotoIndexButton.setEnabled(has_entries)

        if has_entries:
            self.gotoIndexSpin.blockSignals(True)
            self.gotoIndexSpin.setRange(1, total)
            self.gotoIndexSpin.setValue(self.current_index + 1)
            self.gotoIndexSpin.blockSignals(False)

        for widget in (
            self.zoomSlider,
            self.zoomInButton,
            self.zoomOutButton,
            self.zoomFitButton,
        ):
            widget.setEnabled(has_entries)

    def on_zoom_slider_changed(self, value: int) -> None:
        if not self.entries or self._pixmap_item is None:
            return
        clamped = int(np.clip(value, self._ZOOM_MIN, self._ZOOM_MAX))
        scale = clamped / 100.0
        self._manual_zoom = True
        self._fit_pending = False
        self._apply_zoom(scale, update_slider=False)

    def step_zoom(self, delta: int) -> None:
        value = self.zoomSlider.value() + int(delta)
        clamped = int(np.clip(value, self._ZOOM_MIN, self._ZOOM_MAX))
        if clamped != self.zoomSlider.value():
            self.zoomSlider.setValue(clamped)
        else:
            # Even if the slider value didn't change (already at limit), update label.
            self.zoomLabel.setText(f"Масштаб: {int(round(clamped))}%")

    def reset_zoom_to_fit(self) -> None:
        self._manual_zoom = False
        self._fit_pending = True
        self._fit_view_to_pixmap(force=True)

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
        self._update_width_info(entry)
        self._update_resolution_metrics()
        self.refresh_scene()

    def refresh_scene(self):
        if not self.entries:
            return

        entry = self.entries[self.current_index]
        pixmap = QPixmap.fromImage(entry.preview_qimage())

        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            self.scene.addItem(self._pixmap_item)
            self._pixmap_item.setParentItem(self._content_group)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pixmap)

        if self._split_line is None:
            pen = QPen(Qt.red)
            pen.setWidth(4)
            pen.setCosmetic(True)
            self._split_line = QGraphicsLineItem(0, 0, 0, 0)
            self.scene.addItem(self._split_line)
            self._split_line.setParentItem(self._content_group)
            self._split_line.setPen(pen)
            self._split_line.setZValue(5)
        if self._crop_rect_item is None:
            pen = QPen(QColor(255, 215, 0))
            pen.setWidth(4)
            pen.setCosmetic(True)
            self._crop_rect_item = QGraphicsRectItem()
            self.scene.addItem(self._crop_rect_item)
            self._crop_rect_item.setParentItem(self._content_group)
            self._crop_rect_item.setPen(pen)
            self._crop_rect_item.setBrush(QBrush(Qt.NoBrush))
            self._crop_rect_item.setZValue(4)

        self._update_scene_items(entry)
        self._update_grid(entry)
        self._apply_rotation(entry)
        self._update_scene_rect()

        if self._fit_pending and not self._manual_zoom:
            self._fit_view_to_pixmap()

    def _apply_zoom(self, scale: float, *, update_slider: bool = True) -> None:
        scale = float(np.clip(scale, self._ZOOM_MIN / 100.0, self._ZOOM_MAX / 100.0))
        self.view.resetTransform()
        self.view.scale(scale, scale)
        self._zoom = scale
        percent = int(round(scale * 100))
        if update_slider:
            self.zoomSlider.blockSignals(True)
            self.zoomSlider.setValue(percent)
            self.zoomSlider.blockSignals(False)
        self.zoomLabel.setText(f"Масштаб: {percent}%")

    def _fit_view_to_pixmap(self, *, force: bool = False) -> None:
        if self._pixmap_item is None:
            return

        rect = self._pixmap_item.sceneBoundingRect()
        if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return

        viewport = self.view.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            if not force:
                self._fit_pending = True
                QTimer.singleShot(0, lambda: self._fit_view_to_pixmap(force=True))
            return

        scale_x = width / rect.width()
        scale_y = height / rect.height()
        scale = min(scale_x, scale_y)
        if scale <= 0:
            scale = 1.0

        self._fit_pending = False
        self._manual_zoom = False
        self._apply_zoom(scale)

    def _update_scene_items(self, entry: ManualSplitEntry) -> None:
        left = entry.crop_left
        right = entry.crop_right
        top = entry.crop_top
        bottom = entry.crop_bottom

        if self._pixmap_item is not None:
            self._pixmap_item.setOffset(0, 0)

        if self._split_line is not None:
            if entry.split_disabled:
                self._split_line.hide()
            else:
                self._split_line.show()
                self._split_line.setLine(entry.split_x_base, top, entry.split_x_base, bottom)

        if self._crop_rect_item is not None:
            rect = QRectF(left, top, right - left, bottom - top)
            self._crop_rect_item.setRect(rect)

        if not self._handles:
            for side in ("left", "right", "top", "bottom"):
                handle = CropHandle(side)
                handle.moved.connect(lambda value, s=side: self._handle_moved(s, value))
                handle.setParentItem(self._content_group)
                self.scene.addItem(handle)
                self._handles[side] = handle

        min_width = 2
        min_height = 2
        centre_x = (left + right) / 2.0
        centre_y = (top + bottom) / 2.0

        left_handle = self._handles["left"]
        left_handle.blockSignals(True)
        left_handle.set_limits(0, max(0.0, right - min_width), centre_y)
        left_handle.setPos(left, centre_y)
        left_handle.blockSignals(False)
        left_handle.setVisible(right - left > min_width)

        right_handle = self._handles["right"]
        right_handle.blockSignals(True)
        right_handle.set_limits(left + min_width, entry.width, centre_y)
        right_handle.setPos(right, centre_y)
        right_handle.blockSignals(False)
        right_handle.setVisible(right - left > min_width)

        top_handle = self._handles["top"]
        top_handle.blockSignals(True)
        top_handle.set_limits(0, max(0.0, bottom - min_height), centre_x)
        top_handle.setPos(centre_x, top)
        top_handle.blockSignals(False)
        top_handle.setVisible(bottom - top > min_height)

        bottom_handle = self._handles["bottom"]
        bottom_handle.blockSignals(True)
        bottom_handle.set_limits(top + min_height, entry.height, centre_x)
        bottom_handle.setPos(centre_x, bottom)
        bottom_handle.blockSignals(False)
        bottom_handle.setVisible(bottom - top > min_height)

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
        self._update_width_info(entry)

    def _update_width_info(self, entry: ManualSplitEntry) -> None:
        if not self._identical_resolution:
            self.widthInfoLabel.clear()
            self.widthInfoLabel.setVisible(False)
            return

        dims = _entry_page_dimensions(entry)
        text = (
            f"Левая половина: {dims['left_width']}px, правая половина: {dims['right_width']}px"
        )
        self.widthInfoLabel.setText(text)
        self.widthInfoLabel.setVisible(True)

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

        self._update_resolution_metrics()
        metrics = getattr(self, "_resolution_metrics", {}) or {}
        target_width = int(metrics.get("target_page_width") or self._target_page_width or 0)
        target_height = int(metrics.get("target_page_height") or self._target_page_height or 0)
        if target_width <= 0 or target_height <= 0:
            return

        enforce_entry_targets(
            entry,
            target_page_width=target_width,
            target_page_height=target_height,
            target_spread_width=0,
            target_spread_height=0,
        )

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

    @property
    def target_page_width(self) -> int:
        return max(0, int(self._target_page_width))

    @property
    def target_page_height(self) -> int:
        return max(0, int(self._target_page_height))

    @property
    def resolution_metrics(self) -> dict[str, int | str | None]:
        return dict(self._resolution_metrics)

    def set_target_resolution(self, width: int | None, height: int | None) -> None:
        self._target_page_width = int(width or 0)
        self._target_page_height = int(height or 0)
        if self.entries:
            entry = self.entries[self.current_index]
            self._update_split_status(entry)
            self._update_resolution_metrics()

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
            line.setParentItem(self._content_group)
            line.setZValue(3)
            self._grid_lines.append(line)

        for i in range(1, 3):
            y = top + height * i / 3.0
            line = self.scene.addLine(left, y, right, y, pen)
            line.setParentItem(self._content_group)
            line.setZValue(3)
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

    def _update_resolution_metrics(self) -> None:
        if not self._identical_resolution or not self.entries:
            self.heightInfoLabel.setVisible(False)
            self.widthInfoLabel.setVisible(False)
            self.resolutionWarningLabel.clear()
            self.resolutionWarningLabel.setVisible(False)
            self._resolution_metrics = {"total_entries": 0}
            for button in (self.gotoHeightButton, self.gotoWidthButton):
                button.setEnabled(False)
                button.setVisible(False)
            return

        metrics = collect_resolution_metrics(self.entries)
        self._resolution_metrics = metrics

        target_width = int(metrics.get("target_page_width") or 0)
        target_height = int(metrics.get("target_page_height") or 0)

        if target_height > 0:
            self._target_page_height = target_height
        if target_width > 0:
            self._target_page_width = target_width

        for button in (self.gotoHeightButton, self.gotoWidthButton):
            button.setVisible(True)

        height_text = "Максимальная высота рамки: —"
        height_index = metrics.get("max_height_index")
        if target_height > 0:
            if height_index is not None:
                idx = int(height_index)
                if 0 <= idx < len(self.entries):
                    ref_entry = self.entries[idx]
                    number = idx + 1
                    height_text = (
                        f"Максимальная высота рамки: {target_height}px "
                        f"(страница №{number} {ref_entry.relative.name})"
                    )
                    self.gotoHeightButton.setEnabled(True)
                else:
                    self.gotoHeightButton.setEnabled(False)
            else:
                self.gotoHeightButton.setEnabled(False)
        else:
            self.gotoHeightButton.setEnabled(False)

        width_text = "Максимальная ширина рамки: —"
        side = metrics.get("max_width_side")
        index = metrics.get("max_width_index")
        if target_width > 0 and index is not None:
            idx = int(index)
            if 0 <= idx < len(self.entries):
                ref_entry = self.entries[idx]
                number = idx + 1
                width_text = (
                    f"Максимальная ширина {side or ''} страницы: {target_width}px "
                    f"(№{number} {ref_entry.relative.name})"
                )
                self.gotoWidthButton.setEnabled(True)
            else:
                self.gotoWidthButton.setEnabled(False)
        else:
            self.gotoWidthButton.setEnabled(False)

        self.heightInfoLabel.setText(height_text)
        self.widthInfoLabel.setText(width_text)
        self.heightInfoLabel.setVisible(True)
        self.widthInfoLabel.setVisible(True)

        self._update_resolution_warning()

    def _update_resolution_warning(self) -> None:
        metrics = getattr(self, "_resolution_metrics", {}) or {}
        if not metrics:
            self.resolutionWarningLabel.clear()
            self.resolutionWarningLabel.setVisible(False)
            return

        if not self.entries:
            self.resolutionWarningLabel.clear()
            self.resolutionWarningLabel.setVisible(False)
            return

        current_entry = self.entries[self.current_index]
        dims = _entry_page_dimensions(current_entry)
        warnings: list[str] = []

        target_height = int(metrics.get("target_page_height") or 0)
        target_width = int(metrics.get("target_page_width") or 0)
        left_index = metrics.get("max_left_index")
        right_index = metrics.get("max_right_index")

        left_value = int(metrics.get("max_left_width") or 0)
        right_value = int(metrics.get("max_right_width") or 0)

        if target_height > 0 and dims["height"] < target_height:
            if metrics.get("max_height_index") is not None:
                idx = int(metrics["max_height_index"])
                if 0 <= idx < len(self.entries):
                    ref_entry = self.entries[idx]
                    number = idx + 1
                    warnings.append(
                        f"⚠️ Высота рамки {dims['height']}px меньше требуемых {target_height}px "
                        f"(максимум на странице №{number} {ref_entry.relative.name})."
                    )
            else:
                warnings.append(
                    f"⚠️ Высота рамки {dims['height']}px меньше требуемых {target_height}px."
                )

        if not current_entry.split_disabled and target_width > 0:
            if dims["left_width"] < target_width:
                if left_index is not None:
                    idx = int(left_index)
                    if 0 <= idx < len(self.entries):
                        ref_entry = self.entries[idx]
                        number = idx + 1
                        warnings.append(
                            f"⚠️ Левая половина {dims['left_width']}px меньше требуемых {target_width}px "
                            f"(максимум на левой странице №{number} {ref_entry.relative.name}, {left_value}px)."
                        )
                else:
                    warnings.append(
                        f"⚠️ Левая половина {dims['left_width']}px меньше требуемых {target_width}px."
                    )
            if dims["right_width"] < target_width:
                if right_index is not None:
                    idx = int(right_index)
                    if 0 <= idx < len(self.entries):
                        ref_entry = self.entries[idx]
                        number = idx + 1
                        warnings.append(
                            f"⚠️ Правая половина {dims['right_width']}px меньше требуемых {target_width}px "
                            f"(максимум на правой странице №{number} {ref_entry.relative.name}, {right_value}px)."
                        )
                else:
                    warnings.append(
                        f"⚠️ Правая половина {dims['right_width']}px меньше требуемых {target_width}px."
                    )

        if warnings:
            self.resolutionWarningLabel.setText("\n".join(warnings))
        else:
            self.resolutionWarningLabel.clear()
        self.resolutionWarningLabel.setVisible(bool(warnings))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._manual_zoom:
            self._fit_pending = True
            QTimer.singleShot(0, lambda: self._fit_view_to_pixmap(force=True))

    def goto_metric_entry(self, metric: str) -> None:
        metrics = getattr(self, "_resolution_metrics", {}) or {}
        index = None
        if metric == "height":
            index = metrics.get("max_height_index")
        elif metric == "width":
            index = metrics.get("max_width_index")
        if index is None:
            return
        idx = int(index)
        if 0 <= idx < len(self.entries):
            self.current_index = idx
            self.display_current_entry()

    def goto_first(self) -> None:
        if not self.entries:
            return
        self.current_index = 0
        self.display_current_entry()

    def goto_last(self) -> None:
        if not self.entries:
            return
        self.current_index = len(self.entries) - 1
        self.display_current_entry()

    def goto_next(self) -> None:
        if not self.entries:
            return
        if self.current_index < len(self.entries) - 1:
            self.current_index += 1
            self.display_current_entry()

    def goto_previous(self) -> None:
        if not self.entries:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_entry()

    def goto_index(self) -> None:
        if not self.entries:
            return
        value = int(self.gotoIndexSpin.value()) - 1
        value = max(0, min(value, len(self.entries) - 1))
        self.current_index = value
        self.display_current_entry()

    def reset_current(self) -> None:
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.crop_left = 0
        entry.crop_top = 0
        entry.crop_right = entry.width
        entry.crop_bottom = entry.height
        entry.rotation_deg = 0.0
        entry.set_split_base(entry.auto_split_x)
        entry.split_disabled = False
        entry.update_split_from_ratio()
        self._sync_controls(entry)
        self.refresh_scene()

    def toggle_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def adjust_rotation(self, delta: float) -> None:
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.rotation_deg = float(entry.rotation_deg + delta)
        entry.release_image()
        self._sync_controls(entry)
        self.refresh_scene()

    def reset_rotation(self) -> None:
        if not self.entries:
            return
        entry = self.entries[self.current_index]
        entry.rotation_deg = 0.0
        entry.release_image()
        self._sync_controls(entry)
        self.refresh_scene()

    def on_crop_changed(self) -> None:
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

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.fullscreenButton.isChecked():
            self.fullscreenButton.setChecked(False)
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() == Qt.NoModifier:
            previous_index = self.current_index
            self.goto_next()
            event.accept()
            if previous_index == self.current_index:
                return
            return
        super().keyPressEvent(event)

    def _apply_rotation(self, entry: ManualSplitEntry) -> None:
        if self._pixmap_item is None:
            return

        centre_x, centre_y = entry.crop_centre
        self._pixmap_item.setTransformOriginPoint(float(centre_x), float(centre_y))
        self._pixmap_item.setRotation(float(entry.rotation_deg))

        if self._content_group is not None and self._content_group.rotation() != 0.0:
            # Keep overlay geometry unrotated so the operator sees the original
            # detection bounds even while the image preview is rotated.
            self._content_group.setRotation(0.0)

    def _entry_dimensions_text(self, entry: ManualSplitEntry) -> str:
        dims = _entry_page_dimensions(entry)
        return (
            f"Кадр: {dims['height']}px высота, {entry.current_width}px ширина, "
            f"левая {dims['left_width']}px, правая {dims['right_width']}px"
        )

    def _entry_navigation_text(self) -> str:
        return f"{self.current_index + 1}/{len(self.entries)}"

    def closeEvent(self, event):
        if self.fullscreenButton.isChecked():
            self.fullscreenButton.setChecked(False)
        for entry in self.entries:
            try:
                entry.release_image()
            except Exception:
                pass
        super().closeEvent(event)


__all__ = ["CropHandle", "ManualSplitDialog"]
