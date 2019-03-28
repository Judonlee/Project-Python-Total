import tensorflow
import numpy
import os
from DepressionRecognition.Model.DBLSTM_AutoEncoder import DBLSTM_AutoEncoder
from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer

if __name__ == '__main__':
    attention = StandardAttentionInitializer
    attentionScope = 0
    attentionName = 'SA'

    secondAttention = StandardAttentionInitializer
    secondAttentionScope = 0
    secondAttentionName = 'SA_2'

    loadpath = 'E:/ProjectData_Depression/Experiment/DBLSTM-AutoEncoder-Both/%s_%d/' % (attentionName, attentionScope)
    savepath = 'E:/ProjectData_Depression/Experiment/DBLSTM-AutoEncoder-Both/%s_%d_Result/' % (
        attentionName, attentionScope)
    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    # for index in range(len(trainLabel)):
    #     trainLabel[index][0] = float(trainLabel[index][0]) / 24
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))

    classifier = DBLSTM_AutoEncoder(data=trainData, label=trainLabel, seq=trainSeq, regressionWeight=1,
                                    attention=attention, attentionName=attentionName, attentionScope=attentionScope,
                                    secondAttention=secondAttention, secondAttentionScope=secondAttentionScope,
                                    secondAttentionName=secondAttentionName, batchSize=64, learningRate=1E-3)

    for episode in range(88, 89):
        classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
        classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq,
                        logname=savepath + '%04d.csv' % episode)
