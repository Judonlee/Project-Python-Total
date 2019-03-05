from DepressionRecognition.Loader import Load_DBLSTM, Load_EncoderDecoder
from DepressionRecognition.Model.AttentionTransform import AttentionTransform
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    savepath = '/mnt/external/Bobs/201902-Depression/DBLSTM_MCA_10_Second/'
    # os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    trainLabelSeq = None
    # trainData, trainLabel, trainSeq, trainLabelSeq, testData, testLabel, testSeq, testLabelSeq = Load_EncoderDecoder()

    classifier = AttentionTransform(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, trainLabelSeq=trainLabelSeq,
        firstAttention=MonotonicAttentionInitializer, firstAttentionName='MA', firstAttentionScope=10,
        secondAttention=None, secondAttentionName=None, secondAttentionScope=None,
        batchSize=32, learningRate=1E-4, startFlag=True,
        attentionTransformLoss='L1', attentionTransformWeight=100)
    classifier.LoadPart(loadpath=r'E:\ProjectData_Depression\Experiment\EncoderDecoder\MA_10\0019-Network')
    # classifier.EncoderDecoderTrain()
    classifier.Valid()
