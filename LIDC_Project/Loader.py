import numpy


def LoadPCA(name, part, componentNumber=10):
    loadpath = 'E:/ProjectData_LIDC/Features/Step3_PCA/'
    trainData, trainLabel, testData, testLabel = [], [], [], []

    for index in range(5):
        currentData = numpy.load(loadpath + '%s_%d.csv.npy' % (name, index))[:, 0:componentNumber]
        currentLabel = numpy.genfromtxt(fname=loadpath + 'Featurelabel_%d.csv' % index, dtype=int, delimiter=',')
        # print(numpy.shape(currentData), numpy.shape(currentLabel))

        if index == part:
            testData.extend(currentData)
            testLabel.extend(currentLabel)
        else:
            trainData.extend(currentData)
            trainLabel.extend(currentLabel)
    print('Load Completed Part', part, numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData),
          numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel


def LoadDX(name, part, componentNumber=10):
    loadpath = 'E:/ProjectData_LIDC/Features/Step3_DX/'
    trainData, trainLabel, testData, testLabel = [], [], [], []

    for index in range(5):
        currentData = numpy.load(loadpath + '%s_%d.csv.npy' % (name, index))[:, 0:componentNumber]
        currentLabel = numpy.genfromtxt(fname=loadpath + 'Featurelabel_%d.csv' % index, dtype=int, delimiter=',')
        # print(numpy.shape(currentData), numpy.shape(currentLabel))

        if index == part:
            testData.extend(currentData)
            testLabel.extend(currentLabel)
        else:
            trainData.extend(currentData)
            trainLabel.extend(currentLabel)
    print('Load Completed Part', part, numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData),
          numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    LoadDX(name='CurveletFeature', part=1)
