import os
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer
from DepressionRecognition.Loader import Load_DBLSTM, Loader_SentenceLevel, Loader_SpeechLevel
from DepressionRecognition.Model.AttentionTransform_ThreePart import AttentionTransform_ThreePart

if __name__ == '__main__':
    attention = MonotonicAttentionInitializer
    attentionName = 'MA'
    attentionScope = 10
    part = 'frame'
    weight = 10

    savepath = 'E:/ProjectData_Depression/FinalResult_ThreePart/%s_%d_%s_%d/' % (
        attentionName, attentionScope, part, weight)
    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    sentenceTrain, sentenceTest = Loader_SentenceLevel(part='%s-%d-%s' % (attentionName, attentionScope, part))
    speechTrain, speechTest = Loader_SpeechLevel(part='%s-%d-%s' % (attentionName, attentionScope, part))

    classifier = AttentionTransform_ThreePart(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, sentenceLevel=sentenceTrain,
        speechLevel=speechTrain, trainLabelSeq=None, firstAttention=attention, firstAttentionName=attentionName,
        firstAttentionScope=attentionScope, secondAttention=attention, secondAttentionName=attentionName + '_2',
        secondAttentionScope=attentionScope, attentionTransformWeight=weight)
    classifier.Load_AttentionTransform('E:/ProjectData_Depression/Experiment/MA_Both_L1_4096/%04d-Network' % 99)
    # classifier.Train('Test.csv')

    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
