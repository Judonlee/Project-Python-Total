from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    savepath = 'E:/ProjectData_Depression/Experiment/LA1_Both/'
    os.makedirs(savepath)

    trainData, trainSeq, trainLabel, testData, testSeq, testLabel = Load_DBLSTM()
    classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                        firstAttention=LocalAttentionInitializer, secondAttention=LocalAttentionInitializer,
                        firstAttentionScope=1, secondAttentionScope=1,
                        firstAttentionName='LA_1', secondAttentionName='LA_2', graphPath=savepath)
    for episode in range(100):
        print('\nEpisode %d/%d Total Loss = %f' % (
            episode, 100, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
