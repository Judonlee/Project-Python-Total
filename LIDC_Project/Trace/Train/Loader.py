import numpy
import os


def LoadPart(loadpath, appoint):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for counter in range(10):
        data = numpy.load(os.path.join(loadpath, 'Part%d-Data.npy' % counter))
        label = numpy.load(os.path.join(loadpath, 'Part%d-Label.npy' % counter))
        # print(numpy.shape(data), numpy.shape(label))

        if counter == appoint:
            testData.extend(data)
            testLabel.extend(label)
        else:
            trainData.extend(data)
            trainLabel.extend(label)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    trainData, trainLabel, testData, testLabel = LoadPart(loadpath='E:\LIDC\TreatmentTrace\Step7-TotalNpy\OriginCsv',
                                                          appoint=0)
