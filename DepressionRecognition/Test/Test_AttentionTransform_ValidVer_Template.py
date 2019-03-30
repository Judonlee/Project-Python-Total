from DepressionRecognition.Loader import Load_DBLSTM, Load_EncoderDecoder, Load_DBLSTM_ChoosePart
from DepressionRecognition.Model.AttentionTransform import AttentionTransform
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os
import tensorflow

if __name__ == '__main__':
    firstAttention = StandardAttentionInitializer
    firstAttentionName = 'SA'
    firstAttentionScope = None

    secondAttention = None
    secondAttentionName = None
    secondAttentionScope = None

    attentionTransformLoss = 'L1'
    attentionTransformWeight = 100
    lossType = 'MAE'

    loadname = 'Standard'

    loadpath = 'E:/ProjectData_Depression/Experiment/AttentionTransform/%s/%s_%s_%s_%d/' % (
        lossType, firstAttentionName, ('First' if (secondAttention == None) else 'Both'), attentionTransformLoss,
        attentionTransformWeight)
    savepath = 'E:/ProjectData_Depression/Experiment/AttentionTransform/%s/%s_%s_%s_%d_Valid_Result/' % (
        lossType, firstAttentionName, ('First' if (secondAttention == None) else 'Both'), attentionTransformLoss,
        attentionTransformWeight)

    ####################################################################

    os.makedirs(savepath)
    trainData, trainLabel, trainSeq = Load_DBLSTM_ChoosePart(part=['Develop'])
    trainLabelSeq = None
    # trainData, trainLabel, trainSeq, trainLabelSeq, testData, testLabel, testSeq, testLabelSeq = Load_EncoderDecoder()

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = AttentionTransform(
            trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, trainLabelSeq=trainLabelSeq,
            firstAttention=firstAttention, firstAttentionName=firstAttentionName,
            firstAttentionScope=firstAttentionScope,
            secondAttention=secondAttention, secondAttentionName=secondAttentionName,
            secondAttentionScope=secondAttentionScope, attentionTransformLoss=attentionTransformLoss,
            attentionTransformWeight=attentionTransformWeight, lossType=lossType, batchSize=32, learningRate=1E-3,
            startFlag=False)
        for episode in range(99, 100):
            print('\nTreating Episode %d' % episode)
            classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
            classifier.Test(testData=trainData, testLabel=trainLabel, testSeq=trainSeq,
                            logName=savepath + '%04d.csv' % episode)
