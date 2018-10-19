from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 40
    for appoint in range(10):
        savepath = 'D:/ProjectData/Records-Result-CTC-CMU-New-Test/Bands-%d-%d/' % (bands, appoint)
        # if os.path.exists(savepath): continue

        # if not os.path.exists(savepath):
        #     os.makedirs(savepath + 'Decode/')
        #     os.makedirs(savepath + 'Logits/')
        #     os.makedirs(savepath + 'SoftMax/')

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
            loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))

        # exit()
        for trace in range(100):
            if os.path.exists(savepath + 'Decode/Epoch%04d.csv' % trace): continue
            fileDecode = open(savepath + 'Decode/Epoch%04d.csv' % trace, 'w')
            fileLogits = open(savepath + 'Logits/Epoch%04d.csv' % trace, 'w')
            fileSoftMax = open(savepath + 'SoftMax/Epoch%04d.csv' % trace, 'w')
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1, startFlag=False,
                                       batchSize=64)
                classifier.Load(
                    'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New/Bands-%d-%d/%04d-Network'
                    % (bands, appoint, trace))
                matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                       testLabel=testLabel,
                                                                                       testSeq=testSeq)
                print()
                print(matrixDecode)
                print(matrixLogits)
                print(matrixSoftMax)

            for indexX in range(len(matrixDecode)):
                for indexY in range(len(matrixDecode[indexX])):
                    if indexY != 0: fileDecode.write(',')
                    fileDecode.write(str(matrixDecode[indexX][indexY]))
                fileDecode.write('\n')
            for indexX in range(len(matrixLogits)):
                for indexY in range(len(matrixLogits[indexX])):
                    if indexY != 0: fileLogits.write(',')
                    fileLogits.write(str(matrixLogits[indexX][indexY]))
                fileLogits.write('\n')
            for indexX in range(len(matrixSoftMax)):
                for indexY in range(len(matrixSoftMax[indexX])):
                    if indexY != 0: fileSoftMax.write(',')
                    fileSoftMax.write(str(matrixSoftMax[indexX][indexY]))
                fileSoftMax.write('\n')

            fileDecode.close()
            fileLogits.close()
            fileSoftMax.close()
            # exit()
