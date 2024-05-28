# -*- coding: utf-8 -*-
import os


def rename(self):
    count = 0

    for file in os.listdir(self.fileurl):
        count = count + 1
        f = os.path.join(self.fileurl, file)

        if os.path.isfile(f):
            f = f.replace('\\', '/')
            array = f.split("/")
            file = array[-1]
            array = file.split(".")
            nameFile = array[-2]

            newNameFile = count

            if len(str(count)) == 1:
                newNameFile = '00' + str(count)

            if len(str(count)) == 2:
                newNameFile = '0' + str(count)

            replaceName = self.directoryName + "_" + str(newNameFile) + "." + array[-1]

            x = f.replace(nameFile + "." + array[-1], replaceName)
            os.rename(f, x)
        # f = f.replace(nameFile + "." + array[-1], replaceName)
