from DepressionRecognition.Loader import Load_EncoderDecoder
from DepressionRecognition.Model.EncoderDecoder import EncoderDecoder_Base
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    savepath = 'E:/ProjectData_Depression/Experiment/EncoderDecoder/'
    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, trainLabelSeq, testData, testLabel, testSeq, testLabelSeq = Load_EncoderDecoder()

    classifier = EncoderDecoder_Base(
        trainData=trainData, trainLabel=trainLabel, trainDataSeq=trainSeq, trainLabelSeq=trainLabelSeq,
        attention=None, attentionName=None, attentionScope=None, batchSize=32, learningRate=1E-4)
    # classifier.ValidTest()
    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
