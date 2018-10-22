import os
import numpy
import tensorflow
from CTC_Project_Again.Model.E2E_Test import E2E_FinalPooling

if __name__ == '__main__':
    for appoint in range(10):
        loadpath = 'D:/ProjectData/IEMOCAP/OriginVoice-Npy/Appoint-%d/' % appoint
        netpath = 'D:/ProjectData/Project-CTC-Data/Records-E2E-Origin/Appoint-%d/' % appoint

        episode = 99
        trainData = numpy.load(loadpath + 'TrainData.npy')
        trainLabel = numpy.load(loadpath + 'TrainLabel.npy')
        trainSeq = numpy.load(loadpath + 'TrainSeq.npy')
        testData = numpy.load(loadpath + 'TestData.npy')
        testLabel = numpy.load(loadpath + 'TestLabel.npy')
        testSeq = numpy.load(loadpath + 'TestSeq.npy')
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.sum(trainLabel, axis=0),
              numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.sum(testLabel, axis=0))
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = E2E_FinalPooling(trainData=testData, trainLabel=testLabel,
                                          trainSeqLength=testSeq, numClass=4, batchSize=32)
            classifier.Load(loadpath=netpath + '%04d-Network' % episode)
            classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq)
            exit()
