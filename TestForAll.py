from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
import tensorflow
import os
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    for appoint in range(10):
        print('\n\nAppoint', appoint)
        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, startFlag=False, batchSize=64)
            classifier.Load(
                loadpath='D:/ProjectData/Project-CTC-Data/NetworkParameter-CTC-Class5/Bands-%d-%d/WA' % (
                    bands, appoint))
            classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, startFlag=False, batchSize=64)
            classifier.Load(
                loadpath='D:/ProjectData/Project-CTC-Data/NetworkParameter-CTC-Class5/Bands-%d-%d/UA' % (
                    bands, appoint))
            classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
