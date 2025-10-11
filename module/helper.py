# -*- coding: utf-8 -*-
import os
import shutil
import sys
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtGui import QTextCursor, QIcon, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from threading import Thread


def statusLoaded(self, procent):
    self.progressBar.setValue(procent)


def updateLog(self, text):
    cursor = QTextCursor(self.textEdit.document())
    cursor.setPosition(0)
    self.textEdit.setTextCursor(cursor)
    self.textEdit.insertHtml(" "
                             "--------------------------------------"
                             "<br>"
                             "<b>__" + text + "__</b> <br>")
    if text == 'КОНЕЦ ОБРАБОТКИ':
        self.label_8.setText("__ КОНЕЦ ОБРАБОТКИ __")


def prepeaImageEnd(self, text):
    t2 = Thread(target=self.showEndImage, daemon=True)
    t2.start()
    t2.join()


def setParamsUi(self):
    self.dpi = int(self.dpiSpinBox.value())
    self.kf_w = float(self.widthCoefSpinBox.value())
    self.kf_h = float(self.heightCoefSpinBox.value())
    self.width_px = int(self.overlapSpinBox.value())
    self.border_px = int(self.borderSpinBox.value())
    self.pxStartList = int(self.startSearchSpinBox.value())
    self.pxMediumVal = int(self.middleSearchSpinBox.value())
    self.count_cpu = int(self.cpuSpinBox.value())
    self.isRemoveBorder = self.checkBox.isChecked()  # удалять черную рамку
    self.isSplit = self.checkBox_3.isChecked()  # Делить изображение по полам
    self.isManualSplitAdjust = self.checkBox_9.isChecked()  # Ручная корректировка середины
    self.isAddBorder = self.checkBox_2.isChecked()  # Добавлять черную рамку
    self.isAddBorderForAll = self.checkBox_4.isChecked()  # Добавлять черную рамку c 4 сторон
    self.isPxIdentically = self.checkBox_5.isChecked()  # Подстраивать разрешение
    self.isShowStart = self.checkBox_6.isChecked()  # Показывать изначальные сканы
    self.isShowEnd = self.checkBox_7.isChecked()  # Показывать получившмеся сканы
    self.isAddBlackBorder = self.checkBox_8.isChecked()  # Добавлять черную рамку


def getUrl(self):
    directory = QFileDialog.getExistingDirectory(self,
                                                 'Выберите папку со сканами которые необходимо разделить на страницы')
    if directory:
        directory = directory.replace('\\', '/')
        self.fileurl = directory
        _, _, files = next(os.walk(self.fileurl))
        self.countFile = len(files)

        self.label.setText(str(self.countFile))
        dir = os.path.abspath(os.curdir)
        dir = dir.replace('\\', '/')
        self.dirInit = dir + '/' + self.directoryName + self.postfix

        self.label_12.setText(self.fileurl)
        if self.checkBox_6.isChecked():
            print('isChecked')
            t1 = Thread(target=self.showStartImage, daemon=True)
            t1.start()


def Clicked(self, item):
    image = QImage(item.text())
    pixmap = QPixmap.fromImage(image)
    pixmap = pixmap.scaled(int(self.label_8.width() * 0.9), int(self.label_8.height() * 0.9),
                           QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    self.label_8.setPixmap(pixmap)
