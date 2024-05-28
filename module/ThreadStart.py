# -*- coding: utf-8 -*-

from threading import Thread
from PyQt5.QtCore import QThread, pyqtSignal
import os
import shutil


class ThreadStart(QThread):
    sig = pyqtSignal(str)
    log = pyqtSignal(str)
    proc = pyqtSignal(int)
    end = pyqtSignal(str)

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
                 isAddBorder,
                 isAddBorderForAll,
                 isPxIdentically,
                 isShowStart,
                 isShowEnd,
                 isAddBlackBorder,
                 fileurl):
        # Thread.__init__(self)
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

    from module.splitImage import initSplitImage, parseImage
    from module.removeBorder import initRemoveBorder, removeBorder
    from module.removePostBorder import initRemovePostBorder, removePostBorder
    from module.addBorder import initAddBorder, addBorder
    from module.rename import rename
    from module.addListWidget import showStartImage, showEndImage

    def run(self):
        self.log.emit("Основной поток запущен")
        self.process()

    def process(self):
        self.log.emit("Старт обработки")

        dirWithFile = []
        for curdir, subdirs, files in os.walk(self.fileurl):
            if len(subdirs) == 0 and len(files) > 0:
                print("Директория с Файлами: " + format(curdir))
                dirWithFile.append(format(curdir))

            if len(subdirs) > 0 and len(files) == 0:
                print("Директория с папками: " + format(curdir))

        array = self.fileurl.split("/")
        n1 = array[-1]

        if len(dirWithFile) > 1:
            for url in dirWithFile:
                self.fileurl = directory = url.replace('\\', '/')
                array = self.fileurl.split("/")
                self.directoryName = array[-1]

                dir = os.path.abspath(os.curdir)
                dir = dir.replace('\\', '/')
                self.dirInit = dir + '/' + self.directoryName + self.postfix

                dirName = self.fileurl
                listOfFiles = list()
                for (dirpath, dirnames, filenames) in os.walk(dirName):
                    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

                if self.isRemoveBorder:
                    self.log.emit("START УДАЛЕНИЯ РАМКИ")

                    t1 = Thread(target=self.initRemoveBorder, daemon=True)
                    t1.start()
                    t1.join()

                    self.log.emit("STOP УДАЛЕНИЯ РАМКИ")

                    # self.initRemoveBorder()
                    self.fileurl = self.dirInit

                if self.isSplit:
                    self.log.emit("START РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")

                    t1 = Thread(target=self.initSplitImage, daemon=True)
                    t1.start()
                    t1.join()

                    # self.initSplitImage()

                    dir = os.path.abspath(os.curdir)
                    dir = dir.replace('\\', '/')
                    self.fileurl = dir + '/' + self.directoryName

                    self.log.emit("STOP РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")

                if self.isRemoveBorder and self.isAddBlackBorder:
                    self.log.emit("START УДАЛЕНИЯ РАМКИ ДОП")

                    t1 = Thread(target=self.initRemovePostBorder, daemon=True)
                    t1.start()
                    t1.join()

                    self.log.emit("STOP УДАЛЕНИЯ РАМКИ ДОП")
                    # self.initRemovePostBorder()

                if self.isAddBlackBorder:
                    self.log.emit("START ДОБАВЛЕНИЯ РАМКИ")

                    t1 = Thread(target=self.initAddBorder, daemon=True)
                    t1.start()
                    t1.join()

                    self.log.emit("STOP ДОБАВЛЕНИЯ РАМКИ")

                    # self.initAddBorder()
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

            dirName = self.fileurl
            listOfFiles = list()
            for (dirpath, dirnames, filenames) in os.walk(dirName):
                listOfFiles += [os.path.join(dirpath, file) for file in filenames]

            if self.isRemoveBorder:
                self.log.emit("START УДАЛЕНИЯ РАМКИ")

                t1 = Thread(target=self.initRemoveBorder, daemon=True)
                t1.start()
                t1.join()

                self.log.emit("STOP УДАЛЕНИЯ РАМКИ")

                # self.initRemoveBorder()
                self.fileurl = self.dirInit

            if self.isSplit:
                self.log.emit("START РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")

                t1 = Thread(target=self.initSplitImage, daemon=True)
                t1.start()
                t1.join()

                # self.initSplitImage()

                dir = os.path.abspath(os.curdir)
                dir = dir.replace('\\', '/')
                self.fileurl = dir + '/' + self.directoryName

                self.log.emit("STOP РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ")

            if self.isRemoveBorder and self.isAddBlackBorder:
                self.log.emit("START УДАЛЕНИЯ РАМКИ ДОП")

                t1 = Thread(target=self.initRemovePostBorder, daemon=True)
                t1.start()
                t1.join()

                self.log.emit("STOP УДАЛЕНИЯ РАМКИ ДОП")
                # self.initRemovePostBorder()

            if self.isAddBlackBorder:
                self.log.emit("START ДОБАВЛЕНИЯ РАМКИ")

                t1 = Thread(target=self.initAddBorder, daemon=True)
                t1.start()
                t1.join()

                self.log.emit("STOP ДОБАВЛЕНИЯ РАМКИ")

                # self.initAddBorder()
                self.fileurl = self.dirInit

            if self.isRemoveBorder and self.isSplit:
                dir = os.path.abspath(os.curdir)
                dir = dir.replace('\\', '/')
                shutil.rmtree(dir + '/' + self.directoryName + self.postfix)
                self.fileurl = dir + '/' + self.directoryName
                self.rename()

        if self.isShowEnd:
            self.end.emit("STOP")

        self.log.emit("КОНЕЦ ОБРАБОТКИ")
