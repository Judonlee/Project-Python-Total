from LIDC_Project.Records.Loader.LIDC_Loader import LIDC_Loader_Choosed
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from LIDC_Project.Records.External.ROCPrinter import ROCPrinter
import os
import multiprocessing as mp
import time


def Treatment():
    totalRoc, totalName = [], []
    for C in [1, 10, 100, 1000]:
        for gamma in [1e-2, 1e-3, 1e-4]:

            if os.path.exists('C=%d-gamma=%s.csv' % (C, str(gamma))): continue
            file = open('C=%d-gamma=%s.csv' % (C, str(gamma)), 'w')

            rocList = []
            totalName.append([C, gamma])
            for appoint in range(10):
                trainData, trainLabel, testData, testLabel = LIDC_Loader_Choosed(
                    loadpath='D:/ProjectData/LIDC/Npy-Seperate/OriginCsv/', appoint=appoint)

                totalNumber = numpy.sum(testLabel, axis=0)[0]

                trainData = numpy.reshape(trainData, newshape=[-1, 4096])
                testData = numpy.reshape(testData, newshape=[-1, 4096])
                trainLabel = numpy.argmax(trainLabel, axis=1)
                testLabel = numpy.argmax(testLabel, axis=1)
                totalData = numpy.concatenate((trainData, testData), axis=0)
                pca = PCA(n_components=10)
                pca.fit(totalData)

                trainData = pca.transform(trainData)
                testData = pca.transform(testData)

                totalData = numpy.concatenate((trainData, testData), axis=0)
                totalData = scale(totalData)
                trainData = totalData[0:len(trainData)]
                testData = totalData[len(trainData):]
                rocList.append(
                    ROCPrinter(trainData=trainData, trainLabel=trainLabel, testData=testData, testLabel=testLabel,
                               appoint=appoint, C=C, gamma=gamma))
                print(totalName[-1], rocList)
                file.write(str(rocList[-1]) + ',')
                # exit()
            totalRoc.append(rocList)

            print('\n\n')
            print(totalName[-1], totalRoc[-1])
            print('\n\n')
            file.close()

    for index in range(len(totalRoc)):
        print(totalName[index], totalRoc[index])


if __name__ == '__main__':
    threadList = []
    for _ in range(5):
        process = mp.Process(target=Treatment)
        process.start()
        threadList.append(process)
        time.sleep(5)

    for process in threadList:
        process.join()
