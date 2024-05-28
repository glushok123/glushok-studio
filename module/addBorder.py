# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageOps
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


def initAddBorder(self):
    _, _, files = next(os.walk(self.fileurl))
    self.countFile = len(files)
    if not os.path.exists(self.directoryName):
        os.makedirs(self.directoryName)

    dirName = self.fileurl
    listOfFiles = list()

    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    t = ThreadPool(processes=self.count_cpu)
    rs = t.map(self.addBorder, listOfFiles)
    t.close()


def addBorder(self, filename):
    filename = filename.replace('\\', '/')
    filename = filename.replace(str(self.fileurl), '')
    array = filename.split("/")
    urlDir = filename.replace(array[-1], '')

    if not os.path.exists(self.directoryName + urlDir):
        os.makedirs(self.directoryName + urlDir)

    path = self.directoryName
    os.makedirs(path, exist_ok=True)

    color = "black"
    top = self.border_px
    right = self.border_px
    bottom = self.border_px
    left = self.border_px
    border = (int(top), int(right), int(bottom), int(left))

    img = Image.open(self.fileurl + filename)

    new_img = ImageOps.expand(img, border=border, fill=color)
    new_img.save(self.fileurl + filename, dpi=(self.dpi, self.dpi))
