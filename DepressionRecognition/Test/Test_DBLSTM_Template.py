from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
import os
import tensorflow

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    trainData, trainSeq, trainLabel, testData, testSeq, testLabel = Load_DBLSTM()
    loadpath = 'E:/ProjectData_Depression/Experiment/LA1-Both/'
    savepath = 'E:/ProjectData_Depression/Experiment/LA1-Both-Result/'
    os.makedirs(savepath)

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                            firstAttention=LocalAttentionInitializer, secondAttention=LocalAttentionInitializer,
                            firstAttentionScope=1, secondAttentionScope=1,
                            firstAttentionName='LA_1', secondAttentionName='LA_2')
        for episode in range(100):
            print('Treating Episode %d/100' % episode)
            classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
            classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq,
                            logName=savepath + '%04d.csv' % episode)
