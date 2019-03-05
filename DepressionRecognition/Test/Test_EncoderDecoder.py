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
        firstAttention=StandardAttentionInitializer, firstAttentionName='SA', firstAttentionScope=None,
        batchSize=32, learningRate=1E-4, startFlag=True)
    classifier.LoadPart(loadpath=r'E:\ProjectData_Depression\Experiment\EncoderDecoder\Standard\0019-Network')
    print(tensorflow.global_variables())
    for sample in tensorflow.global_variables():
        print(sample)
    print(len(tensorflow.global_variables()))
    classifier.EncoderDecoderTrain()
