import numpy
import tensorflow
from DepressionRecognition.Model.BLSTM_Simple import BLSTM
from DepressionRecognition.Loader import Load_DBLSTM


def Loader(partname):
    loadpath = 'E:/ProjectData_Depression/Experiment/SentenceLevel/%s-Normalization-%s.csv/%04d.csv'

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    trainData, testData = [], []

    for index in range(142):
        data = numpy.genfromtxt(fname=loadpath % (partname, 'Train', index), dtype=float, delimiter=',')
        trainData.append(data)

    for index in range(47):
        data = numpy.genfromtxt(fname=loadpath % (partname, 'Test', index), dtype=float, delimiter=',')
        testData.append(data)

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    usedpart = 'MA-10-frame'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(usedpart)
    numpy.save('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/%s-Train.npy' % usedpart, trainData)
    # numpy.save('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/TrainLabel.npy', trainLabel)
    # numpy.save('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/TrainSeq.npy', trainSeq)

    numpy.save('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/%s-Test.npy' % usedpart, testData)
    # numpy.save('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/TestLabel.npy', testLabel)
    # numpy.save('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/TestSeq.npy', testSeq)
