import os
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer
from DepressionRecognition.Loader import Load_DBLSTM, Loader_SentenceLevel, Loader_SpeechLevel
from DepressionRecognition.Model.AttentionTransform_ThreePart import AttentionTransform_ThreePart

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    AutoEncoderAttention = 'LA'
    AutoEncoderScope = 1

    attention = MonotonicAttentionInitializer
    attentionName = 'MA'
    attentionScope = 10
    part = 'frame'
    weight = 10

    savepath = '/mnt/external/Bobs/201902-Depression/FinalResult_ThreePart_Different/From_%s_%s_%d_%s_%d/' % (
        AutoEncoderAttention, attentionName, attentionScope, part, weight)
    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    sentenceTrain, sentenceTest = Loader_SentenceLevel(part='%s-%d-%s' % (AutoEncoderAttention, AutoEncoderScope, part))
    speechTrain, speechTest = Loader_SpeechLevel(part='%s-%d-%s' % (AutoEncoderAttention, AutoEncoderScope, part))

    classifier = AttentionTransform_ThreePart(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, sentenceLevel=sentenceTrain,
        speechLevel=speechTrain, trainLabelSeq=None, firstAttention=attention, firstAttentionName=attentionName,
        firstAttentionScope=attentionScope, secondAttention=attention, secondAttentionName=attentionName + '_2',
        secondAttentionScope=attentionScope, attentionTransformWeight=weight)
    classifier.Load_AttentionTransform('/mnt/external/Bobs/201902-Depression/MA_Both_L1_4096/%04d-Network' % 99)
    # classifier.Train('Test.csv')

    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
