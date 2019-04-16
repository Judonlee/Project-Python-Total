from DepressionRecognition.Loader import Load_DBLSTM, Loader_SentenceLevel
from DepressionRecognition.Model.DBLSTM_WithHierarchyAutoEncoder import DBLSTM_WithHierarchyAutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    attention = MonotonicAttentionInitializer
    attentionName = 'MA'
    attentionScope = 10
    part = 'frame'

    loadpath = 'E:/ProjectData_Depression/Experiment/DBLSTM_With_Hierarchy/DBLSTM_HA_%s_%d_%s/' % (
        attentionName, attentionScope, part)
    savepath = 'E:/ProjectData_Depression/Experiment/DBLSTM_With_Hierarchy/DBLSTM_HA_%s_%d_%s_Result/' % (
        attentionName, attentionScope, part)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    sentenceTrain, sentenceTest = Loader_SentenceLevel(part='%s-%d-%s' % (attentionName, attentionScope, part))
    classifier = DBLSTM_WithHierarchyAutoEncoder(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, sentenceLevelInformation=sentenceTrain,
        speechLevelInformation=None, firstAttention=attention, secondAttention=attention,
        firstAttentionScope=attentionScope, secondAttentionScope=attentionScope, firstAttentionName=attentionName,
        secondAttentionName=attentionName + '_2', graphPath=savepath, lossType='MAE', startFlag=False)

    for episode in range(99, 100):
        classifier.Load(loadpath + '%04d-Network' % episode)
        classifier.Test(logName=savepath + '%04d.csv' % episode, testData=testData, testLabel=testLabel,
                        testSeq=testSeq, testSentence=sentenceTest, testSpeech=None)
