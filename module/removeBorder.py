# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtGui import QTextCursor, QIcon, QImage
from imutils.perspective import four_point_transform
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from concurrent import futures
import subprocess as subp
import multiprocessing, logging

mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)
import time
from threading import Thread


def initRemoveBorder(self):
    procent = 0
    array = self.fileurl.split("/")
    self.directoryName = array[-1]

    if not os.path.exists(self.directoryName + self.postfix):
        os.makedirs(self.directoryName + self.postfix)

    dirName = self.fileurl
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    countFile = len(listOfFiles)

    print('__СТАРТ УДАЛЕНИЯ РАМКИ__')
    with futures.ThreadPoolExecutor(max_workers=self.count_cpu) as ex:
        ex.map(self.removeBorder, listOfFiles)
        for result in ex.map(self.removeBorder, listOfFiles):
            print(str(result) + " (Удаление черной рамки)")
            procent = procent + 1
            self.proc.emit(int((procent * 100) / countFile))
            #self.statusLoaded()

def removeBorder(self, imageName):
    imageName = imageName.replace('\\', '/')
    imageName = imageName.replace(str(self.fileurl), '')

    try:
        two_conture = False
        imageUrl = self.fileurl + imageName
        stream = open(imageUrl, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        largest_area = 0
        second_area = 0
        l_index = 0
        s_index = 0

        left_x = 0
        right_x = 0
        test_w = 0

        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)

            if area >= second_area:
                if area >= largest_area:
                    second_area = largest_area
                    s_index = l_index

                    largest_area = area
                    l_index = i
                else:
                    second_area = area
                    s_index = i

            c_x, c_y, c_w, c_h = cv2.boundingRect(cnts[i])

            # print('##')
            # print(c_x)
            # print(c_y)
            # print(c_w)
            # print(c_h)
            # print('##')
            if left_x == 0:
                left_x = c_x
            else:
                if 0 < c_x < self.pxStartList:
                    left_x = c_x
                # if (left_x > c_x):
                #    left_x = c_x

            if test_w == 0:
                test_w = c_w
            else:
                if test_w < c_w:
                    test_w = c_w

            c_right_x = c_x + c_w
            if right_x == 0:
                right_x = c_right_x
            else:
                if right_x < c_right_x:
                    right_x = c_right_x

        if largest_area != 0 and second_area != 0 and (largest_area - second_area) < 3000000:
            # print('Обнаружил 2 больших контура')
            x2, y2, w2, h2 = cv2.boundingRect(cnts[s_index])
            two_conture = True
            # cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (229, 11, 11), 2)

        # cv2.drawContours(image, cnts, -1, (0,255,0), 2)
        # cv2.drawContours(image, cnts, l_index, (11, 33, 229), 2)
        # cv2.drawContours(image, cnts, s_index, (229, 11, 11), 2)
        # find the biggest area
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imS = cv2.resize(image, (1000, 1000))
        # cv2.imshow("Result", np.hstack([imS]))

        displayCnt = None

        if len(cnts) > 0:
            for c in cnts:
                peri = cv2.arcLength(c, False)
                approx = cv2.approxPolyDP(c, 0.09 * peri, True)
                if len(approx) == 4:
                    displayCnt = approx
                    break

        points = displayCnt.reshape(4, 2)

        if not two_conture:
            if self.isAddBorder:
                x = x - self.border_px
                w = w + 2 * self.border_px
                y = y - self.border_px
                h = h + 2 * self.border_px

            points[0][0] = x
            points[1][0] = x
            points[2][0] = x + w
            points[3][0] = x + w

            points[0][1] = y
            points[1][1] = y + h
            points[2][1] = y + h
            points[3][1] = y
        else:
            if x < x2:
                if self.isAddBorder:
                    x = x - self.border_px
                    w2 = w2 + self.border_px
                    y = y - self.border_px
                    h = h + 2 * self.border_px

                points[0][0] = x
                points[1][0] = x
                points[2][0] = x2 + w2
                points[3][0] = x2 + w2

                points[0][1] = y
                points[1][1] = y + h
                points[2][1] = y + h
                points[3][1] = y
            else:
                # x2 = left_x

                if self.isAddBorder:
                    x2 = x2 - self.border_px
                    w = w + self.border_px
                    y = y - self.border_px
                    h = h + 2 * self.border_px

                points[0][0] = x2
                points[1][0] = x2
                points[2][0] = x + w
                points[3][0] = x + w

                points[0][1] = y
                points[1][1] = y + h
                points[2][1] = y + h
                points[3][1] = y

        height, width, depth = image.shape

        if points[1][1] < height * self.kf_h or points[3][0] < width * self.kf_w:
            img = Image.open(self.fileurl + imageName)

            new_img = ImageOps.expand(img)
            new_img.save(self.directoryName + self.postfix + imageName, dpi=(self.dpi, self.dpi))
        else:
            warped = four_point_transform(image, points)
            nameFull = self.directoryName + self.postfix + imageName

            cv2.imencode(".jpg", warped)[1].tofile(nameFull)
            cv2.waitKey()

    except:
        img = Image.open(self.fileurl + imageName)
        new_img = ImageOps.expand(img)
        new_img.save(self.directoryName + self.postfix + imageName, dpi=(self.dpi, self.dpi))

        print('Ошибка обработки (файл скопирован в директорию):' + str(imageName))

    return imageName