from LIDC_Project.Loader.LIDC_Loader import LIDC_Loader_Choosed
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from LIDC_Project.External.ROCPrinter import ROCPrinter
from LIDC_Project.External.DXScore import DXFeatureSelection

if __name__ == '__main__':
    totalRoc, totalName = [], []

    for DXNumber in range(5, 200, 5):
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

            totalData = DXFeatureSelection(data=totalData, label=numpy.concatenate((trainLabel, testLabel), axis=0),
                                           maxFeatures=DXNumber)
            print(numpy.shape(totalData))
            exit()
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
