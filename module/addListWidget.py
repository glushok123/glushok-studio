# -*- coding: utf-8 -*-
import os

from PyQt5.QtGui import QTextCursor, QIcon, QImage
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap


def showStartImage(self):
    print('isChecked showStartImage')
    img = []
    dirName = self.fileurl
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for image in listOfFiles:
        url = image.replace('\\', '/')
        url = url.replace(str(self.fileurl), '')
        imageUrl = self.fileurl + url
        img.append(imageUrl)

    for imageUrl in img:
        pixmap = QPixmap(imageUrl)
        pixmap = pixmap.scaledToWidth(200)
        item1 = QListWidgetItem(QIcon(pixmap), imageUrl)
        self.listWidget.setIconSize(QSize(150, 150))
        self.listWidget.addItem(item1)


def showEndImage(self):
    print('isChecked showEndImage')
    img = []
    dirName = self.fileurl
    listOfFiles = list()
    dir = os.path.abspath(os.curdir)
    dir = dir.replace('\\', '/')

    array = self.fileurl.split("/")
    directoryName = array[-1]

    dirName = dir + '/' + directoryName

    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for image in listOfFiles:
        url = image.replace('\\', '/')
        url = url.replace(str(self.fileurl), '')
        #imageUrl = dirName + url
        img.append(url)

    for imageUrl in img:
        pixmap = QPixmap(imageUrl)
        pixmap = pixmap.scaledToWidth(200)
        item1 = QListWidgetItem(QIcon(pixmap), imageUrl)
        self.listWidget_2.setIconSize(QSize(150, 150))
        self.listWidget_2.addItem(item1)
