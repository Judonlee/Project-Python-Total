from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
import tensorflow
from CTC_Project_Again.Model.CTC_CRF_Reuse_BLSTM_NotTrain import CTC_CRF_Reuse
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    appoint = 0
    trace = 99
    netPath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/%04d-Network' \
              % (bands, appoint, trace)
    savepath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CRF-Reuse/Bands-%d-%d/' % (bands, appoint)

    # os.makedirs(savepath)

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))

    for episode in range(100):
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_CRF_Reuse(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, batchSize=64, startFlag=False)
            classifier.Load(loadpath=savepath + '%04d-Network' % episode)

            matrix = classifier.Test_CRF(testData=testData, testLabel=testLabel, testSeq=testSeq)

            file = open('D:/ProjectData/Result/%04d.csv' % episode, 'w')
            for indexA in range(len(matrix)):
                for indexB in range(len(matrix[indexA])):
                    if indexB != 0: file.write(',')
                    file.write(str(matrix[indexA][indexB]))
                file.write('\n')
            file.close()
