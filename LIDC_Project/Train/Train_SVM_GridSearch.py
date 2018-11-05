from LIDC_Project.Loader.LIDC_Loader import LIDC_Loader_Npy
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from LIDC_Project.External.ROCPrinter import ROCPrinter

if __name__ == '__main__':
    totalRoc, totalName = [], []
    for C in [1, 10, 100, 1000]:
        for gamma in [1e-2, 1e-3, 1e-4]:

            rocList = []
            totalName.append([C, gamma])
            for appoint in range(10):
                trainData, trainLabel, testData, testLabel = LIDC_Loader_Npy(
                    loadpath='E:/LIDC/Npy/OriginCsv/Appoint-%d/' % appoint)

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
                               appoint=appoint))
                print(totalName[-1], rocList)
            totalRoc.append(rocList)

            print('\n\n')
            print(totalName[-1], totalRoc[-1])
            print('\n\n')

    for index in range(len(totalRoc)):
        print(totalName[index], totalRoc[index])
