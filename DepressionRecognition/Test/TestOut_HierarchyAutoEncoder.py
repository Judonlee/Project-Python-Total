from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.HierarchyAutoEncoder import HierarchyAutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import numpy
import os

if __name__ == '__main__':
    firstAttention = None
    firstAttentionName = 'None'
    firstAttentionScope = 0
    lossType = 'frame'

    loadpath = 'E:/ProjectData_Depression/HierarchyAutoEncoder/%s-%d-%s/%04d-Parameter' % (
        firstAttentionName, firstAttentionScope, lossType, 99)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(testData),
          numpy.shape(testLabel), numpy.shape(testSeq))

    with open('TrainLabel.csv', 'w') as file:
        for sample in trainLabel: file.write(str(sample[0]) + '\n')
    with open('TestLabel.csv', 'w') as file:
        for sample in testLabel: file.write(str(sample[0]) + '\n')

    classifier = HierarchyAutoEncoder(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName + '_0', firstAttentionScope=firstAttentionScope,
        secondAttention=firstAttention, secondAttentionName=firstAttentionName + '_1',
        secondAttentionScope=firstAttentionScope, lossType=lossType, startFlag=False)

    classifier.Load(loadpath=loadpath)
    # classifier.TestOut(logname='%s-%d-%s-Train.csv' % (firstAttentionName, firstAttentionScope, lossType),
    #                    treatData=trainData, treatSeq=trainSeq, treatname='Second_FinalState')
    # classifier.TestOut(logname='%s-%d-%s-Test.csv' % (firstAttentionName, firstAttentionScope, lossType),
    #                    treatData=testData, treatSeq=testSeq, treatname='Second_FinalState')

    classifier.TestOutMedia(savepath='%s-%d-%s-Train.csv' % (firstAttentionName, firstAttentionScope, lossType),
                            treatData=trainData, treatSeq=trainSeq, treatname='First_FinalState')
    classifier.TestOutMedia(savepath='%s-%d-%s-Test.csv' % (firstAttentionName, firstAttentionScope, lossType),
                            treatData=testData, treatSeq=testSeq, treatname='First_FinalState')

    # classifier.TestOutHuge(
    #     savepath='HierarchyAutoEncoder/SentenceLevel/%s-%d-%s-Train-First/' % (
    #         firstAttentionName, firstAttentionScope, lossType), treatData=trainData, treatSeq=trainSeq,
    #     treatname='First_Output')
    # classifier.TestOutHuge(
    #     savepath='HierarchyAutoEncoder/SentenceLevel/%s-%d-%s-Test-First/' % (
    #         firstAttentionName, firstAttentionScope, lossType), treatData=testData, treatSeq=testSeq,
    #     treatname='First_Output')
