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

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    for index in range(len(trainLabel)):
        trainLabel[index][0] = float(trainLabel[index][0]) / 24
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))

    # trainData, trainSeq = Loader_AutoEncoder()
    # trainLabel = None
    # for sample in trainData:
    #     print(numpy.shape(sample))

    loadpath = 'E:/ProjectData_Depression/Experiment/AutoEncoder-New/%s-%d/' % (attentionName, attentionScope)
    savepath = 'E:/ProjectData_Depression/Experiment/DBLSTM-AutoEncoder/%s-%d/' % (attentionName, attentionScope)
    os.makedirs(savepath)

    classifier = DBLSTM_AutoEncoder(data=trainData, label=trainLabel, seq=trainSeq, attention=attention,
                                    attentionName=attentionName, attentionScope=attentionScope,
                                    secondAttention=secondAttention, secondAttentionScope=secondAttentionScope,
                                    secondAttentionName=secondAttentionName, batchSize=64, learningRate=1E-3)
    classifier.LoadPart(loadpath + '%04d-Network' % 19)
    for episode in range(100):
        print('\nEpisode %d : Total Loss = %f' % (episode, classifier.Train(logname=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
