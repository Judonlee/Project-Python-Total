from LIDC_Project.Records.Loader.LIDC_Loader import LIDC_Loader_Choosed
import numpy
from sklearn.preprocessing import scale
from LIDC_Project.Records.External.ROCPrinter import ROCPrinter
import os
import multiprocessing as mp
import time


def Treatment():
    totalRoc, totalName = [], []
    weights = numpy.load('Weight.npy')

    position = numpy.zeros(4096)
    for counter in range(4096):
        position[numpy.argmax(weights)] = counter
        weights[numpy.argmax(weights)] = 0
    print(position)
    position = position.tolist()
    for DXNumber in range(5, 200, 5):
        if os.path.exists('DX%04d.csv' % DXNumber): continue
        file = open('DX%04d.csv' % DXNumber, 'w')
        rocList = []
        totalName.append(DXNumber)
        for appoint in range(10):
            trainData, trainLabel, testData, testLabel = LIDC_Loader_Choosed(
                loadpath='D:/ProjectData/LIDC/Npy-Seperate/OriginCsv/', appoint=appoint)

            totalNumber = numpy.sum(testLabel, axis=0)[0]

            trainData = numpy.reshape(trainData, newshape=[-1, 4096])
            testData = numpy.reshape(testData, newshape=[-1, 4096])
            trainLabel = numpy.argmax(trainLabel, axis=1)
            testLabel = numpy.argmax(testLabel, axis=1)
            totalData = numpy.concatenate((trainData, testData), axis=0)

            nowTotalData = []
            for index in range(DXNumber):
                if len(nowTotalData) == 0:
                    nowTotalData = [totalData[:, position.index(index)].copy()]
                else:
                    nowTotalData.append(totalData[:, position.index(index)])
            nowTotalData = numpy.transpose(nowTotalData, axes=(1, 0))
            print(numpy.shape(nowTotalData))
            totalData = nowTotalData
            # exit()
            totalData = scale(totalData)
            trainData = totalData[0:len(trainData)]
            testData = totalData[len(trainData):]
            rocList.append(
                ROCPrinter(trainData=trainData, trainLabel=trainLabel, testData=testData, testLabel=testLabel,
                           appoint=appoint))
            print(totalName[-1], rocList)
            file.write(str(rocList[-1]) + ',')
        totalRoc.append(rocList)

        print('\n\n')
        print(totalName[-1], totalRoc[-1])
        print('\n\n')
        file.close()

    for index in range(len(totalRoc)):
        print(totalName[index], totalRoc[index])


if __name__ == '__main__':
    threadList = []
    for _ in range(10):
        process = mp.Process(target=Treatment)
        process.start()
        threadList.append(process)
        time.sleep(5)

    for process in threadList:
        process.join()
