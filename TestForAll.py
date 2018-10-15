from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
import tensorflow
from CTC_Project_Again.Model.CTC_CRF_Reuse_BLSTM_NotTrain import CTC_CRF_Reuse
import os
from pprint import pprint

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    appoint = 0
    savepath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CRF-Reuse/Bands-%d-%d/' % (bands, appoint)

    episode = 99
    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC_CRF_Reuse(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, batchSize=64, startFlag=False)
        for sample in tensorflow.global_variables():
            print(sample)
        exit()
        classifier.Load(loadpath=savepath + '%04d-Network' % episode)
        print('\nEpisode', episode, 'Load Completed\n')
        classifier.Test_LogitsPooling(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
        classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
