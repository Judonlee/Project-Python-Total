from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 120
    for appoint in range(10):
        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
            loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))

        # exit()
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1, startFlag=False,
                                   batchSize=64)
            classifier.Load(
                'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New-Choosed/Bands-%d-%d/UA' % (bands, appoint))
            matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                   testLabel=testLabel,
                                                                                   testSeq=testSeq)
            print(matrixLogits)
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1, startFlag=False,
                                   batchSize=64)
            classifier.Load(
                'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New-Choosed/Bands-%d-%d/WA' % (bands, appoint))
            matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                   testLabel=testLabel,
                                                                                   testSeq=testSeq)
            print(matrixLogits)
