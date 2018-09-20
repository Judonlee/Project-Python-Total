import numpy


def CASIA_Loader(path):
    trainData = numpy.load(path + 'TrainData.npy')
    trainLabel = numpy.load(path + 'TrainLabel.npy') / 63
    developData = numpy.load(path + 'DevelopData.npy')
    developLabel = numpy.load(path + 'DevelopLabel.npy') / 63
    testData = numpy.load(path + 'TestData.npy')
    testLabel = numpy.load(path + 'TestLabel.npy') / 63

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(developData), numpy.shape(developLabel),
          numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, developData, developLabel, testData, testLabel
