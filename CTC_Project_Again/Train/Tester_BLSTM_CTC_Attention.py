import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
from CTC_Project_Again.Model.CTC_BLSTM_Attention import CTC_BLSTM
import os

if __name__ == '__main__':
    bands = 30

    for appoint in range(10):
        loadpath = 'D:/ProjectData/Project-CTC-Data/Records-BLSTM-CTC-Attention/Bands-%d-%d/' % (bands, appoint)
        savepath = 'D:/ProjectData/Project-CTC-Data/Records-Result-BLSTM-CTC-Attention/Bands-%d-%d/' % (bands, appoint)
        if os.path.exists(savepath): continue
        os.makedirs(savepath + 'Decode')
        os.makedirs(savepath + 'Logits')
        os.makedirs(savepath + 'SoftMax')

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))

        for episode in range(100):
            if os.path.exists(savepath + '%04d.csv' % episode):
                continue
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, startFlag=False, batchSize=64)
                print(loadpath + '%04d-Network' % episode)
                classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
                matrixA, matrixB, matrixC = classifier.Test_AllMethods(testData=testData, testLabel=testLabel,
                                                                       testSeq=testSeq)
                print('\n')
                print(matrixA)
                print(matrixB)
                print(matrixC)

                file = open(savepath + 'Decode/%04d.csv' % episode, 'w')
                for indexX in range(len(matrixA)):
                    for indexY in range(len(matrixA[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(matrixA[indexX][indexY]))
                    file.write('\n')
                file.close()

                file = open(savepath + 'Logits/%04d.csv' % episode, 'w')
                for indexX in range(len(matrixB)):
                    for indexY in range(len(matrixB[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(matrixB[indexX][indexY]))
                    file.write('\n')
                file.close()

                file = open(savepath + 'SoftMax/%04d.csv' % episode, 'w')
                for indexX in range(len(matrixC)):
                    for indexY in range(len(matrixC[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(matrixC[indexX][indexY]))
                    file.write('\n')
                file.close()
        # exit()
