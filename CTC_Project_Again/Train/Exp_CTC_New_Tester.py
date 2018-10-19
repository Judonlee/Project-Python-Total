from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_Transcription_Loader_Npy_New
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bands = 30
    appoint = 7
    netpath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU/Bands-' + str(bands) + '-' + str(appoint) + '/'
    savepath = 'D:/ProjectData/Project-CTC-Data/Records-Result-CTC-CMU/Bands-' + str(bands) + '-' + str(appoint) + '/'

    os.makedirs(savepath + 'Decode')
    os.makedirs(savepath + 'Logits')
    os.makedirs(savepath + 'SoftMax')

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
        loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
    dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                             trainSeq=trainSeq, testData=testData,
                                             testLabel=testScription, testSeq=testSeq)
    trainScription, testScription = IEMOCAP_Transcription_Loader_Npy_New(
        loadpath='D:/ProjectData/Project-CTC-Data/IEMOCAP-Transcription-CMU-Npy/Appoint-%d/' % appoint)

    for episode in range(100):
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, learningRate=1e-3, batchSize=64)
            # print(classifier.information)
            classifier.Load(netpath + '%04d-Network' % episode)
            matrixDecode, matrixLogits, matrixSoftMax = \
                classifier.Test_AllMethods(testData=testData, testLabel=testLabel, testSeq=testSeq)
            print(matrixDecode)
            print(matrixLogits)
            print(matrixSoftMax)
            file = open(savepath + 'Decode\\%04d-Result.csv' % episode, 'w')
            for indexX in range(len(matrixDecode)):
                for indexY in range(len(matrixDecode[indexX])):
                    if indexY != 0:
                        file.write(',')
                    file.write(str(matrixDecode[indexX][indexY]))
                file.write('\n')
            file.close()

            file = open(savepath + 'Logits\\%04d-Result.csv' % episode, 'w')
            for indexX in range(len(matrixLogits)):
                for indexY in range(len(matrixLogits[indexX])):
                    if indexY != 0:
                        file.write(',')
                    file.write(str(matrixLogits[indexX][indexY]))
                file.write('\n')
            file.close()

            file = open(savepath + 'SoftMax\\%04d-Result.csv' % episode, 'w')
            for indexX in range(len(matrixSoftMax)):
                for indexY in range(len(matrixSoftMax[indexX])):
                    if indexY != 0:
                        file.write(',')
                    file.write(str(matrixSoftMax[indexX][indexY]))
                file.write('\n')
            file.close()
