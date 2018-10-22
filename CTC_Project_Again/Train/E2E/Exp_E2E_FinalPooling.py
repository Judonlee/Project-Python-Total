import os
import numpy
import tensorflow
from CTC_Project_Again.Model.E2E_Test import E2E_FinalPooling
from CTC_Project_Again.Loader.IEMOCAP_Voice_Loader import VoiceLoader
import matplotlib.pylab as plt

if __name__ == '__main__':
    for appoint in range(10):
        # loadpath = 'D:/ProjectData/IEMOCAP/OriginVoice-Npy/Appoint-%d/' % appoint
        # savepath = 'Records-E2E-Origin/Appoint-%d/' % appoint
        # os.makedirs(savepath)

        # trainData = numpy.load(loadpath + 'TrainData.npy')
        # trainLabel = numpy.load(loadpath + 'TrainLabel.npy')
        # trainSeq = numpy.load(loadpath + 'TrainSeq.npy')
        # testData = numpy.load(loadpath + 'TestData.npy')
        # testLabel = numpy.load(loadpath + 'TestLabel.npy')
        # testSeq = numpy.load(loadpath + 'TestSeq.npy')
        trainData, trainLabel, trainSeq, testData, testLabel, testSeq = VoiceLoader(
            loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Voices/improve/', appoint=appoint)

        # trainData = trainData[0:32]
        # trainLabel = trainLabel[0:32]
        # trainSeq = trainSeq[0:32]
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.sum(trainLabel, axis=0),
              numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.sum(testLabel, axis=0))

        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = E2E_FinalPooling(trainData=testData, trainLabel=testLabel, trainSeqLength=testSeq,
                                          numClass=4)
            for episode in range(100):
                print('\nEpisode %d : Total Loss = %f\n' % (episode, classifier.Train()))
                classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq)
                # classifier.Save(savepath=savepath + '%04d-Network' % episode)
        exit()
