from LIDC_Project.Records.Loader.LIDC_Loader import LIDC_Loader_Choosed
import numpy
from LIDC_Project.Records.External.DXScore import DXSingleCalculation

if __name__ == '__main__':
    appoint = 0
    trainData, trainLabel, testData, testLabel = LIDC_Loader_Choosed(
        loadpath='D:/ProjectData/LIDC/Npy-Seperate/OriginCsv/', appoint=appoint)

    totalNumber = numpy.sum(testLabel, axis=0)[0]
    trainData = numpy.reshape(trainData, newshape=[-1, 4096])
    testData = numpy.reshape(testData, newshape=[-1, 4096])
    trainLabel = numpy.argmax(trainLabel, axis=1)
    testLabel = numpy.argmax(testLabel, axis=1)
    totalData = numpy.concatenate((trainData, testData), axis=0)

    weight = []
    for index in range(4096):
        print('\r%d/4096' % index, end='')
        weight.append(
            DXSingleCalculation(data=totalData[index, :], label=numpy.concatenate((trainLabel, testLabel), axis=0)))
    print(weight)
    numpy.save('Weight.npy', weight)
