from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_Transcription_Loader_Npy_New
from CTC_Project_Again.Model.CTC_BLSTM_Attention import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            savepath = 'Records-BLSTM-CTC-Attention/Bands-' + str(bands) + '-' + str(appoint) + '/'
            # if os.path.exists(savepath): continue
            # os.makedirs(savepath)

            graph = tensorflow.Graph()
            with graph.as_default():
                trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
                    loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))
                # trainScription, testScription = IEMOCAP_Transcription_Loader_Npy_New(
                #     loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU-Npy-Improve/Appoint-%d/' % appoint)
                dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                         trainSeq=trainSeq, testData=testData,
                                                         testLabel=testScription, testSeq=testSeq)
                trainScription, testScription = IEMOCAP_Transcription_Loader_Npy_New(
                    loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU-Npy-Improve/Appoint-%d/' % appoint)
                classifier = CTC_BLSTM(trainData=testData, trainLabel=testScription, trainSeqLength=testSeq,
                                       featureShape=bands, numClass=5, learningRate=1e-3, batchSize=64)
                print(classifier.information)

                for epoch in range(100):
                    print('\rEpoch %d: Total Loss = %f' % (epoch, classifier.Train()))
                    # classifier.Save(savepath=savepath + '%04d-Network' % epoch)
                    matrixA, matrixB, matrixC = classifier.Test_AllMethods(testData=testData, testLabel=testLabel,
                                                                           testSeq=testSeq)
                    print(matrixA)
