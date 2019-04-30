from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.HierarchyAutoEncoder import HierarchyAutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import numpy
import os

if __name__ == '__main__':
    firstAttention = StandardAttentionInitializer
    firstAttentionName = 'SA'
    firstAttentionScope = 0
    lossType = 'sentence'

    loadpath = 'E:/ProjectData_Depression/%s-%d-%s/' % (firstAttentionName, firstAttentionScope, lossType)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(testData),
          numpy.shape(testLabel), numpy.shape(testSeq))

    classifier = HierarchyAutoEncoder(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName + '_0', firstAttentionScope=firstAttentionScope,
        secondAttention=firstAttention, secondAttentionName=firstAttentionName + '_1',
        secondAttentionScope=firstAttentionScope, lossType=lossType, startFlag=False)
    classifier.Load(loadpath + '%04d-Parameter' % 99)

    classifier.Valid()
