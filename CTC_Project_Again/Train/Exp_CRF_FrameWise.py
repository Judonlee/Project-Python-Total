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
            savepath = 'Records-FrameWise-CRF/' + str(bands) + '-' + str(appoint) + '/'
            # if os.path.exists(savepath): continue
            # os.makedirs(savepath)

            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainData = trainData[0:256]
            trainLabel = trainLabel[0:256]
            trainSeq = trainSeq[0:256]
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData,
                                                     trainLabel=FrameWiseLabelTransformation(labels=trainLabel,
                                                                                             seqLen=trainSeq),
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=FrameWiseLabelTransformation(labels=testLabel,
                                                                                            seqLen=testSeq),
                                                     testSeq=testSeq)
            graph = tensorflow.Graph()
            # exit()
            with graph.as_default():
                classifier = CRF_Test(trainData=dataClass.trainData, trainLabel=dataClass.trainLabel,
                                      trainSeqLength=dataClass.trainSeq, featureShape=bands, numClass=5)
                for episode in range(100):
                    Loss = classifier.Train()
                    print('', end='\r')
                    print('Epoch ', episode, ':', Loss)
                    # classifier.Save(savepath=savepath + '%04d-Network' % episode)

                    print('\nTest Part')
                    classifier.Test_CRF(testData=dataClass.trainData, testLabel=dataClass.trainLabel,
                                        testSeq=dataClass.trainSeq)
            exit()
