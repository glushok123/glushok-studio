# -*- coding: utf-8 -*-
import os
import shutil
import sys
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtGui import QTextCursor, QIcon, QImage
from PyQt5.QtWidgets import *
from threading import Thread
from module.ThreadStart import ThreadStart

#pyinstaller --onefile  .\GlushokStudio.py
class MainApp(QMainWindow, Thread):
    def __init__(self):
        super().__init__()
        self.textEdit_8 = None
        self.textEdit_9 = None
        self.action = None
        self.listWidget = None
        self.listWidget_2 = None
        self.checkBox_4 = None
        self.checkBox_5 = None
        self.checkBox_6 = None
        self.checkBox_7 = None
        self.checkBox_8 = None
        self.label_8 = None
        self.label_8 = None
        self.label_8 = None
        self.checkBox_6 = None
        self.textEdit = None
        self.label = None
        self.progressBar = None
        self.label = None
        self.progressBar = None
        self.checkBox_2 = None
        self.checkBox_3 = None
        self.checkBox = None
        self.textEdit_3 = None
        self.textEdit_5 = None
        self.textEdit_6 = None
        self.textEdit_2 = None
        self.textEdit_4 = None
        self.pushButton_2 = None
        self.pushButton = None
        uic.loadUi('gui/index.ui', self)
        self.dpi = ''
        self.countFile = 0
        self.dirInit = ''
        self.width_img = 0
        self.height_img = 0
        self.dirInit = ''
        self.isRemoveBorder = ''
        self.isSplit = ''
        self.isAddBorder = ''
        self.isShowStart = True
        self.isShowEnd = True
        self.isAddBorderForAll = True
        self.isAddBorderForAll = True
        self.isAddBlackBorder = False
        self.isPxIdentically = False
        self.fileurl = ''
        self.width_px = 100
        self.count_cpu = 4
        self.kf_w = 0.6
        self.kf_h = 0.8
        self.pxStartList = 300
        self.pxMediumVal = 100
        self.border_px = 100
        self.arrayErrorFile = []
        self.directoryName = ''
        self.postfix = '(до разделения на страницы)'
        self.procent = 0
        self.countFile = 0
        self.thread = None
        self.worker = None
        self.gif = None
        self.threadStart = None
        self.isUseTread = True
        self.initUI()

    from module.helper import statusLoaded, updateLog, setParamsUi, Clicked, getUrl, prepeaImageEnd
    from module.addListWidget import showStartImage, showEndImage

    def initUI(self):
        self.pushButton.clicked.connect(self.startTread)
        self.pushButton_2.clicked.connect(self.getUrl)
        self.action.triggered.connect(self.getUrl)
        self.listWidget.itemClicked.connect(self.Clicked)
        self.listWidget_2.itemClicked.connect(self.Clicked)

    def ignore_files(self, dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    def startTread(self):
        self.statusLoaded(0)
        path = 'load.gif'
        self.gif = QtGui.QMovie(path)
        self.label_8.setMovie(self.gif)
        self.gif.start()
        self.setParamsUi()

        self.threadStart = ThreadStart("ThreadStart",
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
                                       self.isAddBorder,
                                       self.isAddBorderForAll,
                                       self.isPxIdentically,
                                       self.isShowStart,
                                       self.isShowEnd,
                                       self.isAddBlackBorder,
                                       self.fileurl)

        self.threadStart.log.connect(self.updateLog)
        self.threadStart.end.connect(self.prepeaImageEnd)
        self.threadStart.proc.connect(self.statusLoaded)
        self.threadStart.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec_())
