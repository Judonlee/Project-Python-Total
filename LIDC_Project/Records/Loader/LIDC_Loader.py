import os
import numpy


def LIDC_Loader(nodulePath, nonNodulePath, appoint=0):
    trainData, trainLabel, testData, testLabel = [], [], [], []

    counter = 0
    for indexA in os.listdir(nodulePath):
        print('Loading Nodule :', indexA)
        for indexB in os.listdir(os.path.join(nodulePath, indexA)):
            for indexC in ['Csv']:
                for indexD in os.listdir(os.path.join(nodulePath, indexA, indexB, indexC)):
                    currentData = numpy.genfromtxt(fname=os.path.join(nodulePath, indexA, indexB, indexC, indexD),
                                                   dtype=float, delimiter=',')
                    if counter % 10 == appoint:
                        testData.append(currentData)
                        testLabel.append([1, 0])
                    else:
                        trainData.append(currentData)
                        trainLabel.append([1, 0])
                    counter += 1

    counter = 0
    for indexA in os.listdir(nonNodulePath):
        print('Loading Non-Nodule :', indexA)
        for indexB in os.listdir(os.path.join(nonNodulePath, indexA)):
            currentData = numpy.genfromtxt(fname=os.path.join(nonNodulePath, indexA, indexB), dtype=float,
                                           delimiter=',')
            if counter % 10 == appoint:
                testData.append(currentData)
                testLabel.append([0, 1])
            else:
                trainData.append(currentData)
                trainLabel.append([0, 1])
            counter += 1
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0),
          numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def LIDC_Loader_Another(nodulePath, nonNodulePath, appoint=0):
    trainData, trainLabel, testData, testLabel = [], [], [], []

    counter = 0
    for indexA in os.listdir(nodulePath):
        print('Nodule Loading :', indexA)
        for indexB in os.listdir(os.path.join(nodulePath, indexA)):
            for indexC in os.listdir(os.path.join(nodulePath, indexA, indexB)):
                # print(indexA, indexB, indexC)
                currentData = numpy.genfromtxt(fname=os.path.join(nodulePath, indexA, indexB, indexC), dtype=float,
                                               delimiter=',')
                if counter % 10 == appoint:
                    testData.append(currentData)
                    testLabel.append([1, 0])
                else:
                    trainData.append(currentData)
                    trainLabel.append([1, 0])
                counter += 1

    counter = 0
    for indexA in os.listdir(nonNodulePath):
        print('Non-Nodule Loading :', indexA)
        for indexB in os.listdir(os.path.join(nonNodulePath, indexA)):
            currentData = numpy.genfromtxt(fname=os.path.join(nonNodulePath, indexA, indexB), dtype=float,
                                           delimiter=',')
            if counter % 10 == appoint:
                testData.append(currentData)
                testLabel.append([0, 1])
            else:
                trainData.append(currentData)
                trainLabel.append([0, 1])
            counter += 1
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def LIDC_Loader_Wavelet(nodulePath, nonNodulePath, part, appoint=0):
    trainData, trainLabel, testData, testLabel = [], [], [], []

    counter = 0
    for indexA in os.listdir(nodulePath):
        print('Nodule Loading :', indexA)
        for indexB in os.listdir(os.path.join(nodulePath, indexA)):
            for indexC in os.listdir(os.path.join(nodulePath, indexA, indexB)):
                currentData = numpy.genfromtxt(fname=os.path.join(nodulePath, indexA, indexB, indexC, part),
                                               dtype=float, delimiter=',')
                if counter % 10 == appoint:
                    testData.append(currentData)
                    testLabel.append([1, 0])
                else:
                    trainData.append(currentData)
                    trainLabel.append([1, 0])
                counter += 1

    counter = 0
    for indexA in os.listdir(nonNodulePath):
        print('Non-Nodule Loading :', indexA)
        for indexB in os.listdir(os.path.join(nonNodulePath, indexA)):
            currentData = numpy.genfromtxt(fname=os.path.join(nonNodulePath, indexA, indexB, part), dtype=float,
                                           delimiter=',')
            if counter % 10 == appoint:
                testData.append(currentData)
                testLabel.append([0, 1])
            else:
                trainData.append(currentData)
                trainLabel.append([0, 1])
            counter += 1

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def LIDC_Loader_Npy(loadpath):
    trainData = numpy.load(loadpath + 'TrainData.npy')
    trainLabel = numpy.load(loadpath + 'TrainLabel.npy')
    # print('Train Part Load Completed')
    testData = numpy.load(loadpath + 'TestData.npy')
    testLabel = numpy.load(loadpath + 'TestLabel.npy')
    # print('Test Part Load Completed')
    # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0),
    #       numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def LIDC_Loader_Choosed(loadpath, appoint):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for index in range(10):
        currentData = numpy.load(loadpath + 'Appoint-%d-Data.npy' % index)
        currentLabel = numpy.load(loadpath + 'Appoint-%d-Label.npy' % index)
        print(numpy.shape(currentData), numpy.shape(currentLabel), numpy.sum(currentLabel, axis=0))

        if index == appoint:
            testData.extend(currentData)
            testLabel.extend(currentLabel)
        else:
            trainData.extend(currentData)
            trainLabel.extend(currentLabel)
    print('\n\n')
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def LIDC_NewLoader(loadpath, part):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for choosePart in range(5):
        data = numpy.load(os.path.join(loadpath, 'Part%d-Data.npy' % choosePart))
        label = numpy.load(os.path.join(loadpath, 'Part%d-Label.npy' % choosePart))

        if choosePart == part:
            testData = data
            testLabel = label
        else:
            trainData.extend(data)
            trainLabel.extend(label)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    LIDC_NewLoader(loadpath='D:/LIDC/Origin-Npy/', part=0)
