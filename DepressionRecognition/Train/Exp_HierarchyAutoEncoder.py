from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.HierarchyAutoEncoder import HierarchyAutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import numpy
import os

if __name__ == '__main__':
    firstAttention = MonotonicAttentionInitializer
    firstAttentionName = 'MA'
    firstAttentionScope = 10
    lossType = 'frame'

    savepath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/%s-%d-%s/' % (
        firstAttentionName, firstAttentionScope, lossType)
    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(testData),
          numpy.shape(testLabel), numpy.shape(testSeq))

    classifier = HierarchyAutoEncoder(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName + '_0', firstAttentionScope=firstAttentionScope,
        secondAttention=firstAttention, secondAttentionName=firstAttentionName + '_1',
        secondAttentionScope=firstAttentionScope, lossType=lossType)
    # classifier.Valid()
    for episode in range(100):
        print('\nEpisode %d/100 Total Loss = %f' % (episode, classifier.Train(savepath + '%04d.csv' % episode)))
        classifier.Save(savepath + '%04d-Parameter' % episode)
