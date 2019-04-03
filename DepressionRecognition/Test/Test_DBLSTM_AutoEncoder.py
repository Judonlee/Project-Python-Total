from DepressionRecognition.Loader import Loader_AutoEncoder, Load_DBLSTM
from DepressionRecognition.Model.DBLSTM_AutoEncoder import DBLSTM_AutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import numpy
import os
import tensorflow

if __name__ == '__main__':
    attention = StandardAttentionInitializer
    attentionScope = 0
    attentionName = 'SA'

    secondAttention = StandardAttentionInitializer
    secondAttentionScope = 0
    secondAttentionName = 'SA_2'

    concatType = 'Concat'

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))

    loadpath = 'E:/ProjectData_Depression/Experiment/DBLSTM-AutoEncoder-%s/%s-%d/' % (
        concatType, attentionName, attentionScope)
    savepath = 'E:/ProjectData_Depression/Experiment/DBLSTM-AutoEncoder-%s/%s-%d-Result/' % (
        concatType, attentionName, attentionScope)
    os.makedirs(savepath)

    classifier = DBLSTM_AutoEncoder(data=trainData, label=trainLabel, seq=trainSeq, concatType=concatType,
                                    attention=attention, attentionName=attentionName, attentionScope=attentionScope,
                                    secondAttention=secondAttention, secondAttentionScope=secondAttentionScope,
                                    secondAttentionName=secondAttentionName, batchSize=64, learningRate=1E-3,
                                    startFlag=False)

    # classifier.Train('log.csv')
    for episode in range(99, 100):
        print('Episode', episode)
        classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
        classifier.Test(logname=savepath + '%04d.csv' % episode, testData=testData, testLabel=testLabel,
                        testSeq=testSeq)
