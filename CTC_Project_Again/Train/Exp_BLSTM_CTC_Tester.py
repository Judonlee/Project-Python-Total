from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    savepath = 'D:\\ProjectData\\Records-BLSTM-CTC-Normalized\\Result-Logits\\'

    for bands in [30, 40]:
        for appoint in range(10):
            if os.path.exists(savepath + str(bands) + '-' + str(appoint)): continue
            os.makedirs(savepath + str(bands) + '-' + str(appoint))
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
                loadpath='D:/ProjectData/Project-CTC-Data/Transcription-SingleNumber-Class5/', appoint=appoint)
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=testTranscription, testSeq=testSeq)

            for episode in range(100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                           featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1,
                                           startFlag=False)
                    # print(classifier.information)
                    classifier.Load('D:\\ProjectData\\Records-BLSTM-CTC-Normalized\\Bands-' + str(bands) + '-'
                                    + str(appoint) + '\\%04d-Network' % episode)
                    # classifier.Train()
                    # classifier.Test_SoftMax(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
                    file = open(savepath + str(bands) + '-' + str(appoint) + '\\Epoch%04d.csv' % episode, 'w')
                    matrix = classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
                    for indexX in range(len(matrix)):
                        for indexY in range(len(matrix[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(matrix[indexX][indexY]))
                        file.write('\n')
                    file.close()
                    print()

            # exit()
