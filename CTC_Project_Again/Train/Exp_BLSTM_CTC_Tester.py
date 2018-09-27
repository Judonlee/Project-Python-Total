from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
from CTC_Project_Again.Model.CTC_BLSTM_NN import CTC_BLSTM_NN
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for bands in [30, 40, 60, 80, 100, 120]:
        if bands != 30: continue

        for appoint in range(6):
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
                loadpath='D:/ProjectData/Project-CTC-Data/Transcription-SingleNumber-Class5/', appoint=appoint)
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=testTranscription, testSeq=testSeq)

            episode = 99
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1,
                                       startFlag=False)
                # print(classifier.information)
                classifier.Load('D:\\ProjectData\\Records-BLSTM-CTC-Normalized\\Bands-' + str(bands) + '-'
                                + str(appoint) + '\\%04d-Network' % episode)
                # classifier.Train()
                classifier.Test_SoftMax(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
                classifier.Test_SoftMax(testData=testData, testLabel=testLabel, testSeq=testSeq)
                print()

            # exit()
