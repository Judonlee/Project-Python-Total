import numpy
import os
from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer


def Loader(usedPart):
    loadpath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/SentenceLevel/%s-%s-First/'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    trainData, testData = [], []
    for index in range(142):
        filename = '%04d.npy' % index
        print('Reload Train', filename)
        currentData = numpy.load(os.path.join(loadpath % (usedPart, 'Train'), filename))
        trainData.append(currentData)
        # print(numpy.shape(currentData))

    for index in range(47):
        filename = '%04d.npy' % index
        print('Reload Test', filename)
        currentData = numpy.load(os.path.join(loadpath % (usedPart, 'Test'), filename))
        testData.append(currentData)

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    usedPart = 'LA-1-frame-Normalization'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(usedPart=usedPart)

    attention = LocalAttentionInitializer
    attentionScope = 1
    attentionName = 'LA'

    savepath = 'E:/ProjectData_Depression/DBLSTM_%s_%d/' % (attentionName, attentionScope)
    # os.makedirs(savepath)

    classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                        firstAttention=attention, secondAttention=attention, firstAttentionScope=attentionScope,
                        secondAttentionScope=attentionScope, firstAttentionName=attentionName,
                        secondAttentionName=attentionName + '_2', graphPath=savepath, lossType='RMSE', featureShape=256)
    # classifier.Valid()
    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
