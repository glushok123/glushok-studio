# -*- coding: utf-8 -*-
import os
import shutil
import sys
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtGui import QTextCursor, QIcon, QImage
from PyQt5.QtWidgets import *
from threading import Thread
from module.ThreadStart import ThreadStart
from PyQt5.QtGui import QMovie, QColor
from functools import partial

#pyinstaller --onefile  .\GlushokStudio.py

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        from PyQt5.uic import loadUi
        loadUi('gui/index.ui', self)

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
        self.gif = QMovie('load.gif')
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
        self.threadStart.manualAdjustmentRequested.connect(self.handleManualSplitAdjustment)

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
        from module.splitImage import ManualSplitDialog

        entries = []
        event = None
        if isinstance(payload, dict):
            entries = payload.get('entries') or []
            event = payload.get('event')

        try:
            if not entries:
                return

            dialog = ManualSplitDialog(entries, parent=self)
            result = dialog.exec_()
            if result != QDialog.Accepted:
                for entry in entries:
                    entry.split_x = entry.auto_split_x
        finally:
            if event is not None:
                event.set()

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