from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    savepath = 'E:/ProjectData_Depression/DBLSTM_MCA_10_Both/'
    startPosition = 0
    if startPosition == 0:
        os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                        firstAttention=MonotonicChunkwiseAttentionInitializer,
                        secondAttention=MonotonicChunkwiseAttentionInitializer,
                        firstAttentionScope=10, secondAttentionScope=10,
                        firstAttentionName='MCA_1', secondAttentionName='MCA_2',
                        graphPath=savepath, startFlag=(startPosition == 0), lossType='RMSE')
    if startPosition != 0:
        classifier.Load(savepath + '%04d-Network' % startPosition)
    # classifier.Valid()
    for episode in range(startPosition + 1, 100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
