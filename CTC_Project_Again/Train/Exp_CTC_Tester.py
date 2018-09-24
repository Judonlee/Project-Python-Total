from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC import CTC
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    appoint = 2
    bands = 30
    savepath = 'F:/Project-CTC-Data/Records-CTC-RNN2nd/Bands-' + str(bands) + '-' + str(appoint) + '/'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
        IEMOCAP_Loader(loadpath='F:/Project-CTC-Data/Npy/Bands' + str(bands) + '/', appoint=appoint)
    trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
        loadpath='F:/Project-CTC-Data/Transcription-SingleNumber/', appoint=appoint)
    dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription, trainSeq=trainSeq,
                                             testData=testData, testLabel=testTranscription, testSeq=testSeq)

    traceWA, traceUA = [], []
    for trace in range(100):
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                             featureShape=bands, numClass=6, learningRate=5e-5, rnnLayers=1, startFlag=False)
            classifier.Load('F:\\Project-CTC-Data\\Records-CTC\\Bands-' + str(bands) + '-' + str(
                appoint) + '\\' + '%04d' % trace + '-Network')
            print('Episode %d/100' % trace)
            matrix = classifier.Test_SoftMax(testData=testData, testLabel=testLabel, testSeq=testSeq)
            # exit()

            WA = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3]) / sum(sum(matrix))
            UA = (matrix[0][0] / sum(matrix[0]) + matrix[1][1] / sum(matrix[1]) +
                  matrix[2][2] / sum(matrix[2]) + matrix[3][3] / sum(matrix[3])) / 4
            print(WA, UA)
            traceWA.append(WA)
            traceUA.append(UA)
    for index in range(len(traceWA)):
        print(traceWA[index], ',', traceUA[index])
