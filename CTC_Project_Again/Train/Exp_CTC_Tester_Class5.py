from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    appoint = 1
    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
    savepath = 'D:/ProjectData/Project-CTC-Data/Records-Result-CTC-LR1e-3-RMSP/Bands-%d-%d/' % (bands, appoint)
    if not os.path.exists(savepath): os.makedirs(savepath)
    # exit()

    for trace in range(100):
        if os.path.exists(savepath + 'Epoch%04d.csv' % trace): continue
        file = open(savepath + 'Epoch%04d.csv' % trace, 'w')
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1, startFlag=False,
                                   batchSize=64)
            classifier.Load('D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/%04d-Network'
                            % (bands, appoint, trace))
            matrix = classifier.Test_Decode(testData=testData, testLabel=testLabel, testSeq=testSeq)
            for indexX in range(len(matrix)):
                for indexY in range(len(matrix[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(matrix[indexX][indexY]))
                file.write('\n')
        file.close()
    '''
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
    '''
