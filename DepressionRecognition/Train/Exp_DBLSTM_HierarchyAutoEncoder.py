from DepressionRecognition.Loader import Load_DBLSTM, Loader_SentenceLevel, Loader_SpeechLevel
from DepressionRecognition.Model.DBLSTM_WithHierarchyAutoEncoder import DBLSTM_WithHierarchyAutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    attention = StandardAttentionInitializer
    attentionName = 'SA'
    attentionScope = 0
    part = 'frame'

    savepath = 'E:/ProjectData_Depression/SpeechLevel/DBLSTM_HA_%s_%d_%s/' % (attentionName, attentionScope, part)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    sentenceTrain, speechTrain = None, None

    # sentenceTrain, sentenceTest = Loader_SentenceLevel(part='%s-%d-%s' % (attentionName, attentionScope, part))
    speechTrain, speechTest = Loader_SpeechLevel(part='%s-%d-%s' % (attentionName, attentionScope, part))

    classifier = DBLSTM_WithHierarchyAutoEncoder(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, sentenceLevelInformation=sentenceTrain,
        speechLevelInformation=speechTrain, firstAttention=attention, secondAttention=attention,
        firstAttentionScope=attentionScope, secondAttentionScope=attentionScope, firstAttentionName=attentionName,
        secondAttentionName=attentionName + '_2', graphPath=savepath, lossType='MAE')
    # classifier.Valid()
    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
