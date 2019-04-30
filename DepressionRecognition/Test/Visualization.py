import os
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer
from DepressionRecognition.Loader import Load_DBLSTM, Loader_SentenceLevel, Loader_SpeechLevel
from DepressionRecognition.Model.AttentionTransform_ThreePart import AttentionTransform_ThreePart
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer

if __name__ == '__main__':
    AutoEncoderAttention = 'SA'
    AutoEncoderScope = 0

    attention = StandardAttentionInitializer
    attentionName = 'SA'
    attentionScope = 0
    part = 'sentence'
    weight = 1000

    loadpath = 'E:/ProjectData_Depression/From_%s_%s_%d_%s_%d/' % (
        AutoEncoderAttention, attentionName, attentionScope, part, weight)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    sentenceTrain, sentenceTest = Loader_SentenceLevel(part='%s-%d-%s' % (AutoEncoderAttention, AutoEncoderScope, part))
    speechTrain, speechTest = Loader_SpeechLevel(part='%s-%d-%s' % (AutoEncoderAttention, AutoEncoderScope, part))

    classifier = AttentionTransform_ThreePart(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, sentenceLevel=sentenceTrain,
        speechLevel=speechTrain, trainLabelSeq=None, firstAttention=attention, firstAttentionName=attentionName,
        firstAttentionScope=attentionScope, secondAttention=attention, secondAttentionName=attentionName + '_2',
        secondAttentionScope=attentionScope, attentionTransformWeight=weight, startFlag=False)

    classifier.Load(loadpath=loadpath + '%04d-Network' % 99)
    classifier.Visualization()
