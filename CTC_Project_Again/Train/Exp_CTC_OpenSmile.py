from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_Transcription_Loader_Npy_New
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for appoint in range(10):
        savepath = 'Records-CTC-Class5-Again-LR1E-4/Appoint-' + str(appoint) + '/'
        if os.path.exists(savepath): continue
        os.makedirs(savepath)

        graph = tensorflow.Graph()
        with graph.as_default():
            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Features/GeMAPSv01a-Npy/Appoint-%d/' % appoint)
            trainScription, testScription = IEMOCAP_Transcription_Loader_Npy_New(
                loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU-Npy-Improve/Appoint-%d/' % appoint)
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=testScription, testSeq=testSeq)
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=trainData[0].shape[1], numClass=5, learningRate=1e-3, batchSize=64)
            print(classifier.information)

            for epoch in range(100):
                print('\rEpoch %d: Total Loss = %f' % (epoch, classifier.Train()))
                classifier.Save(savepath=savepath + '%04d-Network' % epoch)
