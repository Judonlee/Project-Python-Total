from shutil import copy
import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/Voice-Target/read/'

    dictionary = {}
    currentData = numpy.genfromtxt(fname=r'D:\ProjectData\MSP-IMPROVE\Dictionary.csv', dtype=str, delimiter=',')
    for sample in currentData:
        dictionary[sample[0]] = sample[1]

    # print(dictionary)

    for indexA in range(1, 7):
        for indexB in os.listdir(os.path.join(loadpath, 'session%d' % indexA)):
            for indexC in os.listdir(os.path.join(loadpath, 'session%d' % indexA, indexB)):
                if indexC != 'R': continue
                for indexD in os.listdir(os.path.join(loadpath, 'session%d' % indexA, indexB, indexC)):
                    label = dictionary[indexD[4:-4]]
                    if label == 'X' or label == 'O': continue
                    gender = indexD.split('-')[3][0]
                    print(gender)

                    if not os.path.exists(os.path.join(savepath, 'Session%d' % indexA, gender, label)):
                        os.makedirs(os.path.join(savepath, 'Session%d' % indexA, gender, label))

                    copy(os.path.join(loadpath, 'session%d' % indexA, indexB, indexC, indexD),
                         os.path.join(savepath, 'Session%d' % indexA, gender, label, indexD))
