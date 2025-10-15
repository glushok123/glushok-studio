# -*- coding: utf-8 -*-
import os
import shutil
import sys
import re
import tempfile
import traceback
import faulthandler
import numpy as np
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtGui import QTextCursor, QIcon, QImage
from PyQt5.QtWidgets import *
from module.ThreadStart import ThreadStart
from PyQt5.QtGui import QMovie, QColor
from functools import partial

try:
    faulthandler.enable()
except Exception:
    # На некоторых сборках PyInstaller stderr может быть недоступен.
    pass

#pyinstaller --onefile  .\GlushokStudio.py


def resource_path(*relative_parts: str) -> str:
    """Resolve resource paths for both source and PyInstaller bundles."""

    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, *relative_parts)


def ensure_ui_is_wellformed(src_path: str) -> str:
    """Return a temporary path to a well-formed copy of the Qt Designer UI file."""
    with open(src_path, encoding='utf-8') as f:
        text = f.read()

    token_pattern = re.compile(r'<(/?)([^\s>/]+)([^>]*)>')
    parts = []
    stack = []  # list of (tag, indent)
    pos = 0

    def append_closing(tag: str, indent: str) -> None:
        if parts and not parts[-1].endswith('\n'):
            parts.append('\n')
        parts.append(f"{indent}</{tag}>")

    for match in token_pattern.finditer(text):
        parts.append(text[pos:match.start()])
        token = match.group(0)

        if token.startswith('<?'):
            parts.append(token)
            pos = match.end()
            continue

        closing = match.group(1) == '/'
        name = match.group(2)
        self_closing = token.endswith('/>')
        start = text.rfind('\n', 0, match.start()) + 1
        indent = text[start:match.start()]

        if closing:
            while stack and stack[-1][0] != name:
                append_closing(*stack.pop())
            if stack and stack[-1][0] == name:
                stack.pop()
                parts.append(token)
            else:
                parts.append(token)
        elif self_closing:
            parts.append(token)
        else:
            stack.append((name, indent))
            parts.append(token)

        pos = match.end()

    parts.append(text[pos:])

    while stack:
        append_closing(*stack.pop())

    fixed_text = ''.join(parts)
    tmp_path = os.path.join(tempfile.gettempdir(), 'glushok_index.ui')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    return tmp_path


def load_ui_with_repair(ui_path: str, baseinstance: QMainWindow) -> None:
    """Load the Qt Designer UI, repairing it only when absolutely necessary."""
    from PyQt5.uic import loadUi
    from xml.etree.ElementTree import ParseError

    try:
        loadUi(ui_path, baseinstance)
        return
    except ParseError:
        # Fall back to a sanitized copy for legacy UI files with mismatched tags.
        repaired_path = ensure_ui_is_wellformed(ui_path)
        loadUi(repaired_path, baseinstance)


def install_global_exception_hook() -> None:
    """Print uncaught exceptions to stderr instead of silently exiting."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        print("[ERROR] Необработанное исключение:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


install_global_exception_hook()

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = resource_path('gui', 'index.ui')
        load_ui_with_repair(ui_path, self)

        # Очередь папок: каждый элемент — словарь {'path': str, 'status': str, 'progress': int}
        self.folderQueue = []
        self.currentIndex = 0
        self.isProcessing = False

        # Параметры обработки (оставил неизменными, как было)
        self.dpi = ''
        self.width_px = 100
        self.count_cpu = 4
        self.kf_w = 0.6
        self.kf_h = 0.8
        self.pxStartList = 300
        self.pxMediumVal = 100
        self.border_px = 100
        self.isRemoveBorder = ''
        self.isSplit = ''
        self.isManualSplitAdjust = False
        self.isAddBorder = ''
        self.isShowStart = True
        self.isShowEnd = True
        self.isAddBorderForAll = True
        self.isAddBlackBorder = False
        self.isPxIdentically = False
        self._activeManualDialog = None

        # Пути к helper-функциям из module.helper и module.addListWidget
        from module.helper import statusLoaded, updateLog, setParamsUi, Clicked, getUrl, prepeaImageEnd
        from module.addListWidget import showStartImage, showEndImage

        # Привязываем методы к экземпляру
        self.statusLoaded = statusLoaded.__get__(self)
        self.updateLog = updateLog.__get__(self)
        self.setParamsUi = setParamsUi.__get__(self)
        self.Clicked = Clicked.__get__(self)
        self.getUrl = getUrl.__get__(self)
        self.prepeaImageEnd = prepeaImageEnd.__get__(self)
        self.showStartImage = showStartImage.__get__(self)
        self.showEndImage = showEndImage.__get__(self)

        self.initUI()

    def initUI(self):
        # Кнопка "ВЫПОЛНИТЬ" теперь запускает очередь
        self.pushButton.clicked.connect(self.startQueueProcessing)

        # Старая логика выбора папки (для одиночного файла) осталась на месте
       # self.pushButton_2.clicked.connect(self.getUrl)
        self.action.triggered.connect(self.getUrl)
        self.listWidget.itemClicked.connect(self.Clicked)
        self.listWidget_2.itemClicked.connect(self.Clicked)

        # Новые кнопки управления очередью
        self.addFolderButton.clicked.connect(self.addFolderToQueue)
        self.removeFolderButton.clicked.connect(self.removeFolderFromQueue)

    def addFolderToQueue(self):
        """
        Открывает QFileDialog с возможностью множественного выбора папок.
        Проверяет содержимое каждой папки и добавляет в очередь только папки с изображениями
        или папки, содержащие другие папки с изображениями.
        """
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)

        # Хак для мультивыбора в QDialog
        view = dialog.findChild(QListView, "listView")
        if view:
            view.setSelectionMode(QAbstractItemView.MultiSelection)
        tree = dialog.findChild(QTreeView)
        if tree:
            tree.setSelectionMode(QAbstractItemView.MultiSelection)

        if dialog.exec_():
            folders = dialog.selectedFiles()
            for folder in folders:
                # Нормализуем путь, заменяя обратные слеши на прямые
                folder = folder.replace('\\', '/')
                
                # Проверяем, нет ли уже этой папки в очереди
                if folder not in [item['path'] for item in self.folderQueue]:
                    # Проверяем содержимое папки
                    valid_folders = self.checkFolderContent(folder)
                    
                    if valid_folders:
                        for valid_folder in valid_folders:
                            # Нормализуем путь для каждой найденной папки
                            valid_folder = valid_folder.replace('\\', '/')
                            if valid_folder not in [item['path'] for item in self.folderQueue]:
                                self.folderQueue.append({'path': valid_folder, 'status': 'Ожидает', 'progress': 0})
                                self.addFolderToListWidget(valid_folder, 'Ожидает')
                    else:
                        QMessageBox.warning(self, "Предупреждение", 
                                          f"Папка {folder} не содержит изображений или папок с изображениями.")

    def checkFolderContent(self, folder_path):
        """
        Проверяет содержимое папки и возвращает список папок, которые содержат изображения.
        
        Args:
            folder_path (str): Путь к проверяемой папке
            
        Returns:
            list: Список путей к папкам, содержащим изображения
        """
        valid_folders = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Нормализуем путь
        folder_path = folder_path.replace('\\', '/')
        
        # Проверяем, содержит ли текущая папка изображения
        has_images = False
        has_subfolders = False
        
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                if os.path.splitext(item.lower())[1] in image_extensions:
                    has_images = True
                    break
            elif os.path.isdir(item_path):
                has_subfolders = True
        
        # Если папка содержит изображения, добавляем её
        if has_images:
            valid_folders.append(folder_path)
        
        # Если папка содержит подпапки, проверяем их
        if has_subfolders:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    subfolders = self.checkFolderContent(item_path)
                    valid_folders.extend(subfolders)
        
        return valid_folders

    def removeFolderFromQueue(self):
        """
        Удаляет из очереди выбранные элементы (по выделенным QListWidgetItem).
        """
        selected = self.folderQueueList.selectedItems()
        for item in selected:
            row = self.folderQueueList.row(item)
            self.folderQueueList.takeItem(row)
            del self.folderQueue[row]

    def addFolderToListWidget(self, folder, status):
        """
        Заводит новый QListWidgetItem с текстом "<путь> — [0%] [Ожидает]" и серым цветом.
        """
        text = f"{folder} — [0%] [{status}]"
        item = QListWidgetItem(text)
        item.setForeground(self.getStatusColor(status))
        self.folderQueueList.addItem(item)

    def updateFolderStatus(self, folder, status, progress=None):
        """
        Обновляет self.folderQueue[i]['status'] и (опционально) ['progress'],
        а затем меняет текст и цвет соответствующего QListWidgetItem.
        """
        for i, entry in enumerate(self.folderQueue):
            if entry['path'] == folder:
                entry['status'] = status
                if progress is not None:
                    entry['progress'] = progress
                text = f"{folder} — [{entry['progress']}%] [{status}]"
                item = self.folderQueueList.item(i)
                item.setText(text)
                item.setForeground(self.getStatusColor(status))
                break

    def getStatusColor(self, status):
        """
        Возвращает QColor в зависимости от статуса.
        """
        colors = {
            'Ожидает': QColor('gray'),
            'Обрабатывается': QColor('blue'),
            'Готово': QColor('green'),
            'Ошибка': QColor('red'),
        }
        return colors.get(status, QColor('black'))

    def startQueueProcessing(self):
        """
        Запускает построчную (последовательную) обработку всех папок из self.folderQueue.
        """
        if self.isProcessing:
            return  # Если уже идёт обработка, игнорируем повторный вызов

        if not self.folderQueue:
            self.updateLog("Очередь пуста.")
            return

        self.currentIndex = 0
        self.isProcessing = True
        self.processNextFolder()

    def processNextFolder(self):
        """
        Запускает ThreadStart для текущего индекса self.currentIndex.
        Когда будет вызван finishFolder, автоматически перейдёт к следующей папке.
        """
        # Если вышли за границы, финишируем всю очередь
        if self.currentIndex >= len(self.folderQueue):
            self.isProcessing = False
            self.updateLog("Обработка очереди завершена.")
            return

        entry = self.folderQueue[self.currentIndex]
        path = entry['path']

        # Помечаем статус "Обрабатывается" и сбрасываем прогресс
        self.updateFolderStatus(path, 'Обрабатывается', 0)

        # Стандартная анимация и установка параметров
        self.statusLoaded(0)
        self.gif = QMovie(resource_path('load.gif'))
        self.label_8.setMovie(self.gif)
        self.gif.start()
        self.setParamsUi()
        self.fileurl = path

        # Создаём и запускаем поток для обработки этой папки
        self.threadStart = ThreadStart(
            "ThreadStart",
            self.dpi,
            self.kf_w,
            self.kf_h,
            self.width_px,
            self.border_px,
            self.pxStartList,
            self.pxMediumVal,
            self.count_cpu,
            self.isRemoveBorder,
            self.isSplit,
            self.isManualSplitAdjust,
            self.isAddBorder,
            self.isAddBorderForAll,
            self.isPxIdentically,
            self.isShowStart,
            self.isShowEnd,
            self.isAddBlackBorder,
            self.fileurl
        )

        # Лог
        self.threadStart.log.connect(self.updateLog)

        # Ручная корректировка разделения
        self.threadStart.manualAdjustmentRequested.connect(
            self.handleManualSplitAdjustment,
        )

        # Сигнал прогресса
        self.threadStart.proc.connect(lambda p, folder=path: self.updateFolderStatus(folder, 'Обрабатывается', p))

        # Сигнал конца
        self.threadStart.end.connect(lambda _, folder=path: self.finishFolder(folder))

        # Обработчик ошибок
        self.threadStart.finished.connect(lambda folder=path: self.checkThreadError(folder))

        # Запускаем поток
        self.threadStart.start()
        self.updateLog(f"Начата обработка папки: {path}")

    def handleManualSplitAdjustment(self, payload):
        from pathlib import Path

        from module.splitImage import (
            ManualSplitDialog,
            ManualSplitEntry,
            _fit_page_to_canvas,
            split_with_fixed_position,
        )
        from module.image_utils import save_with_dpi

        raw_entries = []
        wait_event = None
        result_holder = None
        worker_thread = None
        if isinstance(payload, dict):
            raw_entries = list(payload.get('entries') or [])
            wait_event = payload.get('wait_event')
            result_holder = payload.get('result')
            worker_thread = payload.get('thread')

        def finalize(accepted: bool, width_img: int | None = None, height_img: int | None = None) -> None:
            if result_holder is not None:
                result_holder['accepted'] = bool(accepted)
                if width_img is not None:
                    result_holder['width_img'] = width_img
                if height_img is not None:
                    result_holder['height_img'] = height_img
            if wait_event is not None:
                wait_event.set()

        if not raw_entries:
            finalize(True)
            return

        gui_thread = QtCore.QThread.currentThread()
        app = QApplication.instance()
        app_thread = app.thread() if app else None
        if app_thread is not None and gui_thread is not app_thread:
            QtCore.QTimer.singleShot(0, lambda: self.handleManualSplitAdjustment(payload))
            return

        def reconstruct_entries():
            prepared = []
            for index, data in enumerate(raw_entries, start=1):
                try:
                    source_path = Path(data.get('source_path', ''))
                    if not source_path:
                        raise ValueError('Не указан путь к изображению')
                    width_val = int(data.get('image_width', 0) or 0)
                    height_val = int(data.get('image_height', 0) or 0)
                    if width_val <= 0 or height_val <= 0:
                        raise ValueError('Некорректные размеры изображения')
                    entry = ManualSplitEntry(
                        source_path=source_path,
                        relative=Path(data.get('relative') or source_path.name),
                        target_dir=Path(data.get('target_dir') or '.'),
                        left_path=Path(data.get('left_path', '')),
                        right_path=Path(data.get('right_path', '')),
                        auto_split_x=int(data.get('auto_split_x', 0)),
                        split_x=int(data.get('split_x', data.get('auto_split_x', 0))),
                        overlap=int(data.get('overlap', 0)),
                        crop_left=int(data.get('crop_left', 0)),
                        crop_top=int(data.get('crop_top', 0)),
                        crop_right=int(data.get('crop_right', width_val)),
                        crop_bottom=int(data.get('crop_bottom', height_val)),
                        image_width=width_val,
                        image_height=height_val,
                        rotation_deg=float(data.get('rotation_deg', 0.0)),
                        split_disabled=bool(data.get('split_disabled', False)),
                    )
                except Exception as exc:
                    message = f"[WARN] Не удалось подготовить запись #{index} для ручной корректировки: {exc}"
                    print(message)
                    self.updateLog(message)
                    continue
                prepared.append(entry)
            return prepared

        entries = reconstruct_entries()
        raw_entries.clear()

        if not entries:
            print('[WARN] Нет доступных изображений для ручной корректировки после подготовки')
            finalize(True)
            return

        dpi_value = getattr(worker_thread, 'dpi', self.dpi)
        is_px_identically = bool(getattr(worker_thread, 'isPxIdentically', False))
        width_img = int(getattr(worker_thread, 'width_img', 0) or 0)
        height_img = int(getattr(worker_thread, 'height_img', 0) or 0)

        def open_dialog():
            nonlocal width_img, height_img
            accepted = False
            try:
                print(f"[INFO] Открытие окна ручной корректировки ({len(entries)} элементов)")
                dialog = ManualSplitDialog(
                    entries,
                    parent=self,
                    identical_resolution=is_px_identically,
                    target_page_width=width_img,
                    target_page_height=height_img,
                )
                self._activeManualDialog = dialog
                result = dialog.exec_()
                accepted = result == QDialog.Accepted
                width_img = dialog.target_page_width or width_img
                height_img = dialog.target_page_height or height_img
                if not accepted:
                    for entry in entries:
                        entry.set_split_x(int(entry.auto_split_x))
                    print("[WARN] Пользователь отменил ручную корректировку, применены авто-параметры")

                target_page_width = 0
                target_page_height = 0
                target_spread_width = 0
                target_spread_height = 0

                if is_px_identically and entries:
                    for entry in entries:
                        try:
                            final_image = entry.build_processed_image()
                        except Exception:
                            entry.release_image()
                            continue

                        if entry.split_disabled or final_image.ndim < 2:
                            height_val, width_val = final_image.shape[:2]
                            target_spread_width = max(target_spread_width, width_val)
                            target_spread_height = max(target_spread_height, height_val)
                            entry.release_image()
                            continue

                        final_width = final_image.shape[1]
                        split_position = entry.final_split_position(final_width)
                        split_result = split_with_fixed_position(final_image, split_position, entry.overlap)

                        for page_image in (split_result.left, split_result.right):
                            if page_image is None or page_image.ndim < 2:
                                continue
                            height_val, width_val = page_image.shape[:2]
                            target_page_width = max(target_page_width, width_val)
                            target_page_height = max(target_page_height, height_val)

                        entry.release_image()

                    if target_page_width > 0:
                        dialog.target_page_width = target_page_width
                    if target_page_height > 0:
                        dialog.target_page_height = target_page_height

                    if target_page_width > 0 or target_page_height > 0:
                        for entry in entries:
                            if entry.split_disabled:
                                continue

                            base_position = int(entry.split_x_base)
                            overlap_val = max(0, int(entry.overlap))
                            changed = False

                            current_left_width = base_position - entry.crop_left + overlap_val
                            if target_page_width > 0 and target_page_width > current_left_width:
                                new_left = base_position + overlap_val - target_page_width
                                new_left = int(np.floor(new_left))
                                new_left = max(0, new_left)
                                if new_left < entry.crop_left:
                                    entry.crop_left = new_left
                                    changed = True

                            current_right_width = entry.crop_right - base_position + overlap_val
                            if target_page_width > 0 and target_page_width > current_right_width:
                                new_right = target_page_width - overlap_val + base_position
                                new_right = int(np.ceil(new_right))
                                new_right = min(entry.width, new_right)
                                if new_right > entry.crop_right:
                                    entry.crop_right = max(entry.crop_left + 1, new_right)
                                    changed = True

                            desired_height = min(target_page_height, entry.height) if target_page_height else 0
                            current_height = entry.crop_bottom - entry.crop_top
                            if desired_height > current_height:
                                centre_y = entry.crop_top + current_height / 2.0 if current_height else entry.crop_top
                                half_height = desired_height / 2.0
                                new_top = centre_y - half_height
                                new_bottom = centre_y + half_height
                                if new_top < 0:
                                    new_top = 0
                                    new_bottom = desired_height
                                if new_bottom > entry.height:
                                    new_bottom = entry.height
                                    new_top = entry.height - desired_height

                                new_top_int = int(np.floor(new_top))
                                new_bottom_int = int(np.ceil(new_bottom))
                                if new_bottom_int <= new_top_int:
                                    new_bottom_int = min(entry.height, new_top_int + max(1, int(desired_height)))

                                new_top_int = max(0, new_top_int)
                                new_bottom_int = max(new_top_int + 1, min(entry.height, new_bottom_int))

                                if new_top_int != entry.crop_top or new_bottom_int != entry.crop_bottom:
                                    entry.crop_top = new_top_int
                                    entry.crop_bottom = new_bottom_int
                                    changed = True

                            if changed:
                                entry.set_split_base(base_position)

                max_observed_width = 0
                max_observed_height = 0

                for entry in entries:
                    entry.target_dir.mkdir(parents=True, exist_ok=True)
                    final_image = entry.build_processed_image()

                    if entry.split_disabled:
                        destination = entry.target_dir / entry.relative.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if is_px_identically and final_image is not None and final_image.ndim >= 2:
                            if target_spread_width > 0 and target_spread_height > 0:
                                final_image = _fit_page_to_canvas(
                                    final_image,
                                    target_spread_width,
                                    target_spread_height,
                                )
                        if final_image is not None and final_image.ndim >= 2:
                            height_val, width_val = final_image.shape[:2]
                            max_observed_width = max(max_observed_width, width_val)
                            max_observed_height = max(max_observed_height, height_val)
                        save_with_dpi(final_image, destination, dpi_value)
                        entry.release_image()
                        continue

                    final_width = final_image.shape[1] if final_image.ndim >= 2 else 0
                    split_position = entry.final_split_position(final_width)
                    split_result = split_with_fixed_position(final_image, split_position, entry.overlap)

                    for page_image, page_path in (
                        (split_result.left, entry.left_path),
                        (split_result.right, entry.right_path),
                    ):
                        page_path.parent.mkdir(parents=True, exist_ok=True)
                        if (
                            is_px_identically
                            and page_image is not None
                            and page_image.ndim >= 2
                            and target_page_width > 0
                            and target_page_height > 0
                        ):
                            page_image = _fit_page_to_canvas(
                                page_image,
                                target_page_width,
                                target_page_height,
                            )
                        if page_image is not None and page_image.ndim >= 2:
                            height_val, width_val = page_image.shape[:2]
                            max_observed_width = max(max_observed_width, width_val)
                            max_observed_height = max(max_observed_height, height_val)
                        save_with_dpi(page_image, page_path, dpi_value)

                    entry.release_image()

                if is_px_identically:
                    width_img = max(width_img, max_observed_width)
                    height_img = max(height_img, max_observed_height)

                if worker_thread is not None:
                    worker_thread.width_img = width_img
                    worker_thread.height_img = height_img
            except Exception as exc:
                error_text = f"Ошибка при обработке ручной корректировки: {exc}"
                self.updateLog(error_text)
                tb = traceback.format_exc()
                print(error_text, file=sys.stderr)
                print(tb, file=sys.stderr)
                accepted = False
            finally:
                self._activeManualDialog = None
                for entry in entries:
                    try:
                        entry.release_image()
                    except Exception:
                        pass
                finalize(accepted, width_img, height_img)

        QtCore.QTimer.singleShot(0, open_dialog)

    def finishFolder(self, path):
        """
        Вызывается, когда поток для папки path успешно завершился.
        Меняем статус на "Готово" и запускаем следующий.
        """
        self.updateFolderStatus(path, 'Готово', 100)
        self.currentIndex += 1
        self.processNextFolder()

    def checkThreadError(self, path):
        """
        Этот слот вызывается, когда QThread полностью завершил работу.
        Проверим, если статус по-прежнему "Обрабатывается" (а поток не вызвал finishFolder),
        то значит была ошибка, и ставим "Ошибка".
        """
        # Найдём запись в очереди
        for entry in self.folderQueue:
            if entry['path'] == path:
                if entry['status'] == 'Обрабатывается':
                    # Поток завершился без вызова finishFolder → ошибка
                    self.updateFolderStatus(path, 'Ошибка', entry['progress'])
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec_())

