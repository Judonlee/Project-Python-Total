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


if __name__ == '__main__':
    # for appoint in range(10):
    #     savepath = 'E:/LIDC/Npy/LBP_P=24_R=3/Appoint-%d/' % appoint
    #     os.makedirs(savepath)
    #
    #     trainData, trainLabel, testData, testLabel = LIDC_Loader_Another(
    #         nodulePath='E:/LIDC/LIDC-Nodules-LBP/Result_P=24_R=3/Csv/',
    #         nonNodulePath='E:/LIDC/LIDC-NonNodules-LBP/Result_P=24_R=3/Csv/', appoint=appoint)
    #     numpy.save(savepath + 'TrainData.npy', trainData)
    #     numpy.save(savepath + 'TrainLabel.npy', trainLabel)
    #     numpy.save(savepath + 'TestData.npy', testData)
    #     numpy.save(savepath + 'TestLabel.npy', testLabel)
    for appoint in range(10):
        savepath = 'E:/LIDC/Npy/Wavelet-db2/Appoint-%d/' % appoint
        os.makedirs(savepath)
        for part in ['cA.csv', 'cD.csv', 'cH.csv', 'cV.csv']:
            trainData, trainLabel, testData, testLabel = LIDC_Loader_Wavelet(
                nodulePath='E:/LIDC/LIDC-Nodule-Wavelet/db2/Csv/',
                nonNodulePath='E:/LIDC/LIDC-NonNodule-Wavelet/db2/Csv/',
                part=part, appoint=appoint)
            numpy.save(savepath + part[0:part.find('.')] + '-Train.npy', trainData)
            numpy.save(savepath + part[0:part.find('.')] + '-Test.npy', testData)
            numpy.save(savepath + 'TrainLabel.npy', trainLabel)
            numpy.save(savepath + 'TestLabel.npy', testLabel)
