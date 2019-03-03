import numpy
import tensorflow
import os
from DepressionRecognition.Model.AttentionTransform import AttentionTransform
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
from DepressionRecognition.Loader import Load_EncoderDecoder

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, trainLabelSeq, testData, testLabel, testSeq, testLabelSeq = Load_EncoderDecoder()

    classifier = AttentionTransform(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, trainLabelSeq=trainLabelSeq,
        firstAttention=LocalAttentionInitializer, firstAttentionName='LA', firstAttentionScope=1,
        batchSize=32, learningRate=1E-4, startFlag=False)
    classifier.Load(loadpath=r'E:\ProjectData_Depression\Experiment\EncoderDecoder\LA_1\0019-Network')
    print(tensorflow.global_variables())
    print(len(tensorflow.global_variables()))
    classifier.Train()
