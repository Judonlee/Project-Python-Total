import numpy
import os
from DepressionRecognition.Model.BLSTM_Simple import BLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer


def Loader(usedPart):
    trainData = numpy.load('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/%s-Train.npy' % usedPart)
    trainLabel = numpy.reshape(numpy.load('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/TrainLabel.npy'),
                               [-1])

    testData = numpy.load('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/%s-Test.npy' % usedPart)
    testLabel = numpy.reshape(numpy.load('E:/ProjectData_Depression/Experiment/SentenceLevel/Npy/TestLabel.npy'), [-1])

    trainSeq, testSeq = [], []
    for sample in trainData:
        trainSeq.append(len(sample))
    for sample in testData:
        testSeq.append(len(sample))

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    attention = MonotonicAttentionInitializer
    attentionName = 'MA'
    attentionScope = 10
    usedPart = '%s-%d-frame' % (attentionName, attentionScope)
    os.makedirs(usedPart)
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(usedPart=usedPart)

    classifier = BLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, attention=attention,
                       attentionName=attentionName, attentionScope=attentionScope)
    for episode in range(100):
        print('\nTrain Episode %d Loss = %f' % (episode, classifier.Train()))
        classifier.Test(logname=usedPart + '/%04d.csv' % episode, testData=testData, testLabel=testLabel,
                        testSeq=testSeq)
