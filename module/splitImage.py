# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtGui import QTextCursor, QIcon, QImage
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from concurrent import futures

def initSplitImage(self):
    _, _, files = next(os.walk(self.fileurl))


    if not os.path.exists(self.directoryName):
        os.makedirs(self.directoryName)

    listOfFiles = list()

    for (dirpath, dirnames, filenames) in os.walk(self.fileurl):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    procent = 0
    countFile = len(listOfFiles)

    print('__СТАРТ РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ__')
    with futures.ThreadPoolExecutor(max_workers=self.count_cpu) as ex:
        ex.map(self.removeBorder, listOfFiles)
        for result in ex.map(self.parseImage, listOfFiles):
            print(str(result) + " (Разделение по страницам)")
            procent = procent + 1
            self.proc.emit(int((procent * 100) / countFile))

def parseImage(self, filename):
    filename = filename.replace('\\', '/')
    filename = filename.replace(str(self.fileurl), '')
    array = filename.split("/")
    urlDir = filename.replace(array[-1], '')
    if not os.path.exists(self.directoryName + urlDir):
        os.makedirs(self.directoryName + urlDir)

    path = self.directoryName
    os.makedirs(path, exist_ok=True)

    withImageMediumPx = 0

    img = cv2.imdecode(np.fromfile(self.fileurl + filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    height, width, depth = img.shape

    if height > width:
        width_cutoff = width
        number = (filename.split(".")[0])
        s1 = img[:, :width_cutoff]
        pathSave = path + '/' + number + '.jpg'
        cv2.imencode(".jpg", s1)[1].tofile(pathSave)
    else:
        imgP = cv2.imdecode(np.fromfile(self.fileurl + filename, dtype=np.uint8), 0)

        # imgP = cv2.imread(self.fileurl + filename, 0)
        closed = cv2.morphologyEx(imgP, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
        dens = np.sum(imgP, axis=0)
        mean = np.mean(dens)
        thresh = 255 * np.ones_like(img)
        k = 0.9

        center = width / 2
        for idx, val in enumerate(dens):
            if val < k * mean:
                if withImageMediumPx == 0:
                    withImageMediumPx = idx

                if center - self.pxMediumVal < idx < center + self.pxMediumVal and idx - withImageMediumPx > 1000:
                    withImageMediumPx = idx

        width_cutoff = width // 2

        if withImageMediumPx < center - self.pxMediumVal or withImageMediumPx > center + self.pxMediumVal:
            s1 = img[:, :width_cutoff + self.width_px + 1]
            s2 = img[:, width_cutoff - self.width_px:]
        else:
            s1 = img[:, :withImageMediumPx + self.width_px + 1]
            s2 = img[:, withImageMediumPx - self.width_px:]

        # s1 = img[:, :width_cutoff + self.width_px + 1]
        # s1 = img[:, :withImageMediumPx + self.width_px + 1]
        # s2 = img[:, width_cutoff - self.width_px:]
        # s2 = img[:, withImageMediumPx - self.width_px:]

        number = (filename.split(".")[0])
        number1 = number + '_1'
        number2 = number + '_2'

        pathSave1 = path + '/' + number1 + '.jpg'
        pathSave2 = path + '/' + number2 + '.jpg'

        cv2.imencode(".jpg", s1)[1].tofile(pathSave1)
        cv2.imencode(".jpg", s2)[1].tofile(pathSave2)

        img1 = Image.open(pathSave1)
        img2 = Image.open(pathSave2)

        new_img1 = ImageOps.expand(img1)
        new_img1.save(pathSave1, dpi=(self.dpi, self.dpi))

        new_img2 = ImageOps.expand(img2)
        new_img2.save(pathSave2, dpi=(self.dpi, self.dpi))

        imgt1 = cv2.imdecode(np.fromfile(pathSave1, dtype=np.uint8),
                             cv2.IMREAD_UNCHANGED)
        height, width, depth = imgt1.shape

        imgt1 = Image.open(pathSave1)
        if self.width_img == 0:
            self.width_img = int(width)
            self.height_img = int(height)

        if self.isPxIdentically:
            imgt1 = imgt1.resize((self.width_img, self.height_img), Image.Resampling.LANCZOS)

        new_img = ImageOps.expand(imgt1)
        new_img.save(pathSave1, dpi=(self.dpi, self.dpi))

        imgt1 = cv2.imdecode(np.fromfile(pathSave2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        height, width, depth = imgt1.shape

        imgt1 = Image.open(pathSave2)

        if self.width_img == 0:
            self.width_img = int(width)
            self.height_img = int(height)

        if self.isPxIdentically:
            imgt1 = imgt1.resize((self.width_img, self.height_img), Image.Resampling.LANCZOS)

        new_img = ImageOps.expand(imgt1)
        new_img.save(pathSave2, dpi=(self.dpi, self.dpi))

    return filename