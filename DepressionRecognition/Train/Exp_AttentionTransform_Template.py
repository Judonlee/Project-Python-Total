from DepressionRecognition.Loader import Load_DBLSTM, Load_EncoderDecoder
from DepressionRecognition.Model.AttentionTransform import AttentionTransform
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os

if __name__ == '__main__':
    firstAttention = StandardAttentionInitializer
    firstAttentionName = 'SA'
    firstAttentionScope = None

    secondAttention = None
    secondAttentionName = None
    secondAttentionScope = None

    attentionTransformLoss = 'L1'
    attentionTransformWeight = 100000
    lossType = 'RMSE'

    loadname = 'Standard'

    savepath = 'E:/ProjectData_Depression/Experiment/AttentionTransform/%s/%s_%s_%s_%d/' % (
        lossType, firstAttentionName, ('First' if (secondAttention == None) else 'Both'), attentionTransformLoss,
        attentionTransformWeight)

    ####################################################################

    os.makedirs(savepath)
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    trainLabelSeq = None
    # trainData, trainLabel, trainSeq, trainLabelSeq, testData, testLabel, testSeq, testLabelSeq = Load_EncoderDecoder()

    classifier = AttentionTransform(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, trainLabelSeq=trainLabelSeq,
        firstAttention=firstAttention, firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope,
        secondAttention=secondAttention, secondAttentionName=secondAttentionName,
        secondAttentionScope=secondAttentionScope, attentionTransformLoss=attentionTransformLoss,
        attentionTransformWeight=attentionTransformWeight, lossType=lossType, batchSize=32, learningRate=1E-3,
        startFlag=True)
    classifier.LoadPart(loadpath='E:/ProjectData_Depression/Experiment/EncoderDecoder/%s/0019-Network' % loadname)
    # classifier.EncoderDecoderTrain()

    for episode in range(100):
        print('\nEpisode %d/100 Total Loss = %f' % (episode, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
