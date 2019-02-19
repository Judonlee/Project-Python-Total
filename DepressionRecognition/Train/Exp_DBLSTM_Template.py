from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    savepath = 'E:/ProjectData_Depression/Experiment/DBLSTM_MA_First/'
    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                        firstAttention=MonotonicAttentionInitializer, secondAttention=None,
                        firstAttentionScope=8, secondAttentionScope=None,
                        firstAttentionName='MA_1', secondAttentionName=None, graphPath=savepath)
    # classifier.Valid()
    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
