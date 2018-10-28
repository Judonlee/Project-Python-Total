import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Model.CRF_BLSTM_Test import CRF_BLSTM
import os

if __name__ == '__main__':
    bands = 80
    for appoint in [4, 5]:
        savepath = 'D:/ProjectData/Project-CTC-Data/Records-CRF-BLSTM-Improve-Choosed-UA/Bands-' + str(
            bands) + '-' + str(appoint) + '/'

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))
        trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
            loadpath='D:/ProjectData/Project-CTC-Data/CTC-SeqLabel-Class5-Improve-Choosed-UA/Bands-%d-%d/' %
                     (bands, appoint))

        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainSeqLabel, trainSeq=trainSeq,
                                                 testData=testData, testLabel=testSeqLabel, testSeq=testSeq)

        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CRF_BLSTM(trainData=dataClass.trainData, trainLabel=dataClass.trainLabel,
                                   trainSeqLength=dataClass.trainSeq, featureShape=bands, numClass=4,
                                   learningRate=1e-3, batchSize=64)
            print(classifier.information)
            # if appoint == 8: startPosition = 77
            # if appoint == 9: startPosition = 69
            # if appoint == 7: startPosition = 96
            # if appoint == 9: startPosition = 72
            if appoint == 4: startPosition = 96
            if appoint == 5: startPosition = 72
            classifier.Load(loadpath=savepath + '%04d-Network' % startPosition)
            # classifier.Train()
            # exit()
            for episode in range(startPosition + 1, 100):
                loss = classifier.Train()
                print('\rEpisode %04d : Loss = %f' % (episode, loss))
                # classifier.Test_Decode(testData=trainData, testLabel=trainSeqLabel, testSeq=trainSeq)
                classifier.Save(savepath=savepath + '%04d-Network' % episode)
            # exit()
        # exit()
