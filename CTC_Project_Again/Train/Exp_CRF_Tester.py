import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader
from __Base.DataClass import DataClass_TrainTest_Sequence
import numpy
from CTC_Project_Again.Model.CRF_BLSTM_Test import CRF_Test
import os


def FrameWiseLabelTransformation(labels, seqLen):
    result = []
    for index in range(len(labels)):
        current = numpy.ones(seqLen[index]) * (numpy.argmax(numpy.array(labels[index])) + 1)
        result.append(current)
    return result


if __name__ == '__main__':
    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            loadpath = 'Records-FrameWise-CRF/' + str(bands) + '-' + str(appoint) + '/'

            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainData = trainData[0:32]
            trainLabel = trainLabel[0:32]
            trainSeq = trainSeq[0:32]
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData,
                                                     trainLabel=FrameWiseLabelTransformation(labels=trainLabel,
                                                                                             seqLen=trainSeq),
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=FrameWiseLabelTransformation(labels=testLabel,
                                                                                            seqLen=testSeq),
                                                     testSeq=testSeq)
            print(dataClass.trainLabel)
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CRF_Test(trainData=dataClass.trainData, trainLabel=dataClass.trainLabel,
                                      trainSeqLength=dataClass.trainSeq, featureShape=bands, numClass=5,
                                      startFlag=False)
                classifier.Load(loadpath=loadpath + '0099-Network')
                classifier.Test_CRF(testData=dataClass.trainData, testLabel=dataClass.trainLabel,
                                    testSeq=dataClass.trainSeq)
            exit()
