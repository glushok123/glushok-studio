# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread, pyqtSignal
import os
import shutil


class ThreadStart(QThread):
    sig = pyqtSignal(str)
    log = pyqtSignal(str)
    proc = pyqtSignal(int)
    end = pyqtSignal(str)
    manualAdjustmentRequested = pyqtSignal(object)

    def __init__(self,
                 name,
                 dpi,
                 kf_w,
                 kf_h,
                 width_px,
                 border_px,
                 pxStartList,
                 pxMediumVal,
                 count_cpu,
                 isRemoveBorder,
                 isSplit,
                 isManualSplitAdjust,
                 isAddBorder,
                 isAddBorderForAll,
                 isPxIdentically,
                 isShowStart,
                 isShowEnd,
                 isAddBlackBorder,
                 fileurl):
        super(ThreadStart, self).__init__()
        self.name = name
        self.dpi = dpi
        self.kf_w = kf_w
        self.kf_h = kf_h
        self.width_px = width_px
        self.border_px = border_px
        self.pxStartList = pxStartList
        self.pxMediumVal = pxMediumVal
        self.count_cpu = count_cpu
        self.isRemoveBorder = isRemoveBorder
        self.isSplit = isSplit
        self.isManualSplitAdjust = isManualSplitAdjust
        self.isAddBorder = isAddBorder
        self.isAddBorderForAll = isAddBorderForAll
        self.isPxIdentically = isPxIdentically
        self.isShowStart = isShowStart
        self.isShowEnd = isShowEnd
        self.isAddBlackBorder = isAddBlackBorder
        self.fileurl = fileurl
        self.directoryName = None
        self.postfix = '(до разделения на страницы)'
        self.width_img = 0
        self.height_img = 0
        self.original_fileurl = fileurl  # ← сохраним исходный путь отдельно

    from module.splitImage import initSplitImage, parseImage
    from module.removeBorder import initRemoveBorder, removeBorder
    from module.removePostBorder import initRemovePostBorder, removePostBorder
    from module.addBorder import initAddBorder, addBorder
    from module.rename import rename
    from module.addListWidget import showStartImage, showEndImage

    def run(self):
        try:
            self.log.emit("Основной поток запущен")
            self.log.emit(f"[DEBUG] Параметры обработки:")
            self.log.emit(f"[DEBUG] DPI: {self.dpi}")
            self.log.emit(f"[DEBUG] Коэффициенты: w={self.kf_w}, h={self.kf_h}")
            self.log.emit(f"[DEBUG] Ширина: {self.width_px}")
            self.log.emit(f"[DEBUG] Граница: {self.border_px}")
            self.log.emit(f"[DEBUG] Удаление рамки: {self.isRemoveBorder}")
            self.log.emit(f"[DEBUG] Разделение: {self.isSplit}")
            self.log.emit(f"[DEBUG] Добавление рамки: {self.isAddBorder}")
            self.log.emit(f"[DEBUG] Путь к файлу: {self.fileurl}")
            
            self.process()

            import time

            base_path = os.path.abspath(os.curdir)
            self.log.emit(f"DEBUG: запускаем переименование в {base_path}")

            try:
                for name in os.listdir(base_path):
                    full_path = os.path.join(base_path, name)
                    self.log.emit(f"Проверка: {name}")

                    if not os.path.isdir(full_path):
                        self.log.emit(f"Пропущено (не папка): {name}")
                        continue

                    if '(до разделения на страницы)' not in name:
                        self.log.emit(f"Пропущено (нет шаблона): {name}")
                        continue

                    # Определяем новое имя и путь
                    new_name = name.replace(' (до разделения на страницы)', '').replace('(до разделения на страницы)',
                                                                                        '')
                    new_path = os.path.join(base_path, new_name)

                    self.log.emit(f"Найдено: {name} → {new_name}")

                    if os.path.exists(new_path):
                        # Проверяем, пуста ли целевая папка
                        if not os.listdir(new_path):
                            self.log.emit(f"Целевая папка {new_name} пуста. Перемещаем содержимое...")
                            try:
                                for item in os.listdir(full_path):
                                    shutil.move(os.path.join(full_path, item), new_path)
                                os.rmdir(full_path)
                                self.log.emit(f"Перенос завершён. Удалена папка: {name}")
                            except Exception as e:
                                self.log.emit(f"Ошибка при переносе в {new_name}: {str(e)}")
                        else:
                            self.log.emit(f"Пропущено: {new_name} уже существует и не пуста")
                    else:
                        try:
                            os.rename(full_path, new_path)
                            self.log.emit(f"Переименовано: {name} → {new_name}")
                        except Exception as e:
                            self.log.emit(f"Ошибка при переименовании {name}: {str(e)}")

            except Exception as e:
                self.log.emit(f"ОШИБКА ПРИ ПЕРЕИМЕНОВАНИИ: {str(e)}")

            # --- Блок переноса результата в структуру родителя ---
            try:
                base_path = os.path.abspath(os.curdir)
                self.log.emit(f"[DEBUG] Текущая директория: {base_path}")
                self.log.emit(f"[DEBUG] Исходный путь: {self.original_fileurl}")
                
                original_parts = os.path.normpath(self.original_fileurl).split(os.sep)
                self.log.emit(f"[DEBUG] Разбитый путь: {original_parts}")
                
                # Убираем букву диска из пути, если она есть
                if len(original_parts) > 0 and len(original_parts[0]) == 2 and original_parts[0][1] == ':':
                    original_parts = original_parts[1:]
                
                # Находим индекс папки с файлами (последняя папка в пути)
                last_folder_index = len(original_parts) - 1
                self.log.emit(f"[DEBUG] Индекс последней папки: {last_folder_index}")
                
                # Берем максимум 2 уровня вложенности от папки с файлами
                if last_folder_index >= 2:
                    # Если есть как минимум 2 уровня выше, берем их
                    sub_path_parts = original_parts[last_folder_index-2:last_folder_index+1]
                elif last_folder_index >= 1:
                    # Если есть только 1 уровень выше, берем его и текущую папку
                    sub_path_parts = original_parts[last_folder_index-1:last_folder_index+1]
                else:
                    # Если нет уровней выше, берем только текущую папку
                    sub_path_parts = original_parts[last_folder_index:last_folder_index+1]
                
                self.log.emit(f"[DEBUG] Выбранные части пути: {sub_path_parts}")

                # Создаем путь для результата рядом с программой
                result_dir = os.path.join(base_path, *sub_path_parts)
                # Путь к исходной папке с результатом
                source_dir = os.path.join(base_path, self.directoryName)

                self.log.emit(f"[DEBUG] Путь к исходной папке: {source_dir}")
                self.log.emit(f"[DEBUG] Путь к папке результата: {result_dir}")
                self.log.emit(f"[DEBUG] Существует ли исходная папка: {os.path.exists(source_dir)}")
                self.log.emit(f"[DEBUG] Существует ли папка результата: {os.path.exists(result_dir)}")

                if os.path.abspath(source_dir) == os.path.abspath(result_dir):
                    self.log.emit(f"[DEBUG] Итоговая папка уже на месте: {result_dir}")
                else:
                    # Создаем структуру папок для результата
                    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
                    self.log.emit(f"[DEBUG] Создана структура папок: {os.path.dirname(result_dir)}")
                    
                    # Перемещаем содержимое исходной папки в новую структуру
                    if os.path.exists(result_dir):
                        self.log.emit(f"[DEBUG] Удаляем существующую папку результата: {result_dir}")
                        shutil.rmtree(result_dir)
                    
                    self.log.emit(f"[DEBUG] Перемещаем из {source_dir} в {result_dir}")
                    shutil.move(source_dir, result_dir)
                    self.log.emit(f"[DEBUG] Проверяем результат после перемещения: {os.path.exists(result_dir)}")
                    self.log.emit(f"Итоговая папка перемещена в: {result_dir}")
            except Exception as e:
                self.log.emit(f"Ошибка при переносе итоговой папки: {str(e)}")
                import traceback
                self.log.emit(f"[DEBUG] Полный стек ошибки: {traceback.format_exc()}")

            self.log.emit("КОНЕЦ ОБРАБОТКИ")
            self.end.emit("STOP")  # ← теперь всегда вызывается, даже при isShowEnd = False
        except Exception as e:
            self.log.emit(f"ОШИБКА В run(): {str(e)}")
            self.end.emit("ERROR")

    def process(self):
        self.log.emit("Старт обработки")

        dirWithFile = []
        for curdir, subdirs, files in os.walk(self.fileurl):
            if len(subdirs) == 0 and len(files) > 0:
                print("Директория с Файлами: " + format(curdir))
                dirWithFile.append(format(curdir))

            if len(subdirs) > 0 and len(files) == 0:
              #  print("Директория с папками: " + format(curdir))
                u = format(curdir)
                u = u.replace('\\', '/')
                array = u.split("/")
                dir = array[-1]



       # print(dirWithFile)
        array = self.fileurl.split("/")
        n1 = array[-1]
        if not os.path.exists(n1):
            os.makedirs(n1)

        if len(dirWithFile) > 0:
            for url in dirWithFile:
                self.fileurl = url.replace('\\', '/')
                array = self.fileurl.split("/")
                startSave = False
                for nameDir in array:
                    if startSave:
                        self.directoryName =  self.directoryName + '/' + nameDir
                    if nameDir == n1:
                        startSave = True
                        index = array.index(nameDir)
                        self.directoryName = nameDir


                lenArray = len(array)

                dir = os.path.abspath(os.curdir)
                dir = dir.replace('\\', '/')
                self.dirInit = dir + '/' + self.directoryName + self.postfix

                dirName = self.fileurl
                listOfFiles = list()
                for (dirpath, dirnames, filenames) in os.walk(dirName):
                    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

                if self.isRemoveBorder:
                    self.log.emit("START УДАЛЕНИЯ РАМКИ")
                    self.initRemoveBorder()
                    self.log.emit("STOP УДАЛЕНИЯ РАМКИ")
                    self.fileurl = self.dirInit

                if self.isSplit:
                    self.log.emit("START РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")
                    self.initSplitImage()

                    dir = os.path.abspath(os.curdir)
                    dir = dir.replace('\\', '/')
                    self.fileurl = dir + '/' + self.directoryName

                    self.log.emit("STOP РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")

                if self.isRemoveBorder and self.isAddBlackBorder:
                    self.log.emit("START УДАЛЕНИЯ РАМКИ ДОП")
                    self.initRemovePostBorder()
                    self.log.emit("STOP УДАЛЕНИЯ РАМКИ ДОП")

                if self.isAddBlackBorder:
                    self.log.emit("START ДОБАВЛЕНИЯ РАМКИ")
                    self.initAddBorder()
                    self.log.emit("STOP ДОБАВЛЕНИЯ РАМКИ")
                    self.fileurl = self.dirInit

                if self.isRemoveBorder and self.isSplit:
                    dir = os.path.abspath(os.curdir)
                    dir = dir.replace('\\', '/')
                    shutil.rmtree(dir + '/' + self.directoryName + self.postfix)
                    self.fileurl = dir + '/' + self.directoryName
                    self.rename()

        else:
            array = self.fileurl.split("/")
            self.directoryName = array[-1]

            dir = os.path.abspath(os.curdir)
            dir = dir.replace('\\', '/')
            self.dirInit = dir + '/' + self.directoryName + self.postfix

            print("Директория с папками 2: " + format(self.directoryName))

            if not os.path.exists(self.directoryName):
                os.makedirs(self.directoryName)

            dirName = self.fileurl
            listOfFiles = list()
            for (dirpath, dirnames, filenames) in os.walk(dirName):
                listOfFiles += [os.path.join(dirpath, file) for file in filenames]

            if self.isRemoveBorder:
                self.log.emit("START УДАЛЕНИЯ РАМКИ")
                self.initRemoveBorder()
                self.log.emit("STOP УДАЛЕНИЯ РАМКИ")
                self.fileurl = self.dirInit

            if self.isSplit:
                self.log.emit("START РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")
                self.initSplitImage()

                dir = os.path.abspath(os.curdir)
                dir = dir.replace('\\', '/')
                self.fileurl = dir + '/' + self.directoryName

                self.log.emit("STOP РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")

            if self.isRemoveBorder and self.isAddBlackBorder:
                self.log.emit("START УДАЛЕНИЯ РАМКИ ДОП")
                self.initRemovePostBorder()
                self.log.emit("STOP УДАЛЕНИЯ РАМКИ ДОП")

            if self.isAddBlackBorder:
                self.log.emit("START ДОБАВЛЕНИЯ РАМКИ")
                self.initAddBorder()
                self.log.emit("STOP ДОБАВЛЕНИЯ РАМКИ")
                self.fileurl = self.dirInit

            if self.isRemoveBorder and self.isSplit:
                dir = os.path.abspath(os.curdir)
                dir = dir.replace('\\', '/')
                shutil.rmtree(dir + '/' + self.directoryName + self.postfix)
                self.fileurl = dir + '/' + self.directoryName
                self.rename()



        # if self.isShowEnd:
       #     self.end.emit("STOP")

        self.log.emit("КОНЕЦ ОБРАБОТКИ")
        self.log.emit("DEBUG: дошёл до конца process()")
