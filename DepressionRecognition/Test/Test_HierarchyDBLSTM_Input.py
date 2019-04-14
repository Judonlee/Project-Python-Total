import numpy
import os
from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer


def Loader(usedPart):
    loadpath = 'D:/GitHub/DepressionRecognition/Test/HierarchyAutoEncoder/SentenceLevel/%s-%s-First/'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    trainData, testData = [], []
    for index in range(142):
        filename = '%04d.npy' % index
        print('Reload Train', filename)
        currentData = numpy.load(os.path.join(loadpath % (usedPart, 'Train'), filename))
        currentData = numpy.concatenate([currentData[0], currentData[1]], axis=2)
        trainData.append(currentData)
        print(numpy.shape(currentData))

    for index in range(47):
        filename = '%04d.npy' % index
        print('Reload Test', filename)
        currentData = numpy.load(os.path.join(loadpath % (usedPart, 'Test'), filename))
        currentData = numpy.concatenate([currentData[0], currentData[1]], axis=2)
        testData.append(currentData)
        print(numpy.shape(currentData))

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    attention = LocalAttentionInitializer
    attentionScope = 1
    attentionName = 'LA'
    usedPart = '%s-%d-sentence' % (attentionName, attentionScope)
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(usedPart=usedPart)

    loadpath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/%s/' % usedPart
    savepath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/%s_Result/' % usedPart
    # os.makedirs(savepath)

    classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                        firstAttention=attention, secondAttention=attention, firstAttentionScope=attentionScope,
                        secondAttentionScope=attentionScope, firstAttentionName=attentionName,
                        secondAttentionName=attentionName + '_2', graphPath=savepath, lossType='RMSE', featureShape=256,
                        startFlag=False)
    # classifier.Valid()
    for episode in range(99, 100):
        classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
        classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq,
                        logName=savepath + '%04d.csv' % episode)
