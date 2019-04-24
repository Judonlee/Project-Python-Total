from DepressionRecognition.Loader import Load_DBLSTM, Loader_SentenceLevel, Loader_SpeechLevel
from DepressionRecognition.Model.DBLSTM_WithHierarchyAutoEncoder import DBLSTM_WithHierarchyAutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    autoencoderName = 'LA'
    autooencoderScope = 1
    part = 'frame'

    attention = MonotonicAttentionInitializer
    attentionName = 'MA'
    attentionScope = 10

    loadpath = 'E:/ProjectData_Depression/DBLSTM_With_Hierarchy_SentenceTarget/DBLSTM_HA_From_%s_To_%s_%d_%s/' % (
        autoencoderName, attentionName, attentionScope, part)
    savepath = 'E:/ProjectData_Depression/DBLSTM_With_Hierarchy_SentenceTarget/DBLSTM_HA_From_%s_To_%s_%d_%s_Result/' % (
        autoencoderName, attentionName, attentionScope, part)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    sentenceTrain, sentenceTest, speechTrain, speechTest = None, None, None, None

    sentenceTrain, sentenceTest = Loader_SentenceLevel(part='%s-%d-%s' % (autoencoderName, autooencoderScope, part))
    speechTrain, speechTest = Loader_SpeechLevel(part='%s-%d-%s' % (autoencoderName, autooencoderScope, part))

    classifier = DBLSTM_WithHierarchyAutoEncoder(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, sentenceLevelInformation=sentenceTrain,
        speechLevelInformation=speechTrain, firstAttention=attention, secondAttention=attention,
        firstAttentionScope=attentionScope, secondAttentionScope=attentionScope, firstAttentionName=attentionName,
        secondAttentionName=attentionName + '_2', graphPath=savepath, lossType='MAE', startFlag=False)

    for episode in range(99, 100):
        classifier.Load(loadpath + '%04d-Network' % episode)

        # classifier.Train(logName='log.csv')
        classifier.Test(logName=savepath + '%04d.csv' % episode, testData=testData, testLabel=testLabel,
                        testSeq=testSeq, testSentence=sentenceTest, testSpeech=speechTest)
