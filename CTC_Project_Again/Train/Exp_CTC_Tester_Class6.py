from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    for appoint in range(10):
        savepath = 'Results-LogitsPooling/Part' + str(appoint) + '/'
        # if os.path.exists(savepath): continue
        # os.makedirs(savepath)
        trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
            IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                           appoint=appoint)
        trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
            loadpath='D:/ProjectData/Project-CTC-Data/Transcription-IntersectionWordNumber-Class6/', appoint=appoint)
        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription, trainSeq=trainSeq,
                                                 testData=testData, testLabel=testTranscription, testSeq=testSeq)

        traceWA, traceUA = [], []
        for trace in range(100):
            if os.path.exists(savepath + '%04d.csv' % trace): continue
            file = open(savepath + '%04d.csv' % trace, 'w')
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=6, learningRate=5e-5, rnnLayers=1, startFlag=False)
                classifier.Load(
                    'D:\\ProjectData\\Project-CTC-Data\\Records-CTC-Class6\\Bands-' + str(bands) + '-' + str(
                        appoint) + '\\' + '%04d' % trace + '-Network')
                # print('\n\n\nLoading Completed')
                # continue
                print('Episode %d/100' % trace)
                matrix = classifier.Test_LogitsPooling_Class6(testData=testData, testLabel=testLabel, testSeq=testSeq)
                # exit()

                WA = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3]) / sum(sum(matrix))
                UA = (matrix[0][0] / sum(matrix[0]) + matrix[1][1] / sum(matrix[1]) +
                      matrix[2][2] / sum(matrix[2]) + matrix[3][3] / sum(matrix[3])) / 4
                print(WA, UA)
                traceWA.append(WA)
                traceUA.append(UA)

            file.write(str(WA) + ',' + str(UA))
            file.close()
        for index in range(len(traceWA)):
            print(traceWA[index], ',', traceUA[index])
