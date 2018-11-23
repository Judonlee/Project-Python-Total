from LIDC_Project.Trace.Train.Tools import LoadPart, DXSingleCalculation
import numpy
import os

if __name__ == '__main__':
    used = 'OriginCsv'

    trainData, trainLabel, testData, testLabel = LoadPart(loadpath='E:/LIDC/TreatmentTrace/Step7-TotalNpy/%s/' % used,
                                                          appoint=-1)
    trainData = numpy.reshape(trainData, [-1, numpy.shape(trainData)[1] * numpy.shape(trainData)[2]])
    trainLabel = numpy.argmax(trainLabel, axis=1)

    weightList = []
    for index in range(numpy.shape(trainData)[1]):
        print(index)
        weightList.append(DXSingleCalculation(data=trainData[:, index], label=trainLabel))
    print(weightList)

    with open('WeightResult.csv', 'w') as file:
        for index in range(len(weightList)):
            if index != 0:
                file.write(',')
            file.write(str(weightList[index]))
