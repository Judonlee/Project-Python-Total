from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_Transcription_Loader_Npy_New
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    conf = 'eGeMAPSv01a'

    for appoint in range(10):
        netpath = 'D:/ProjectData/Records-OpenSmile/eGeMAPSv01a-Appoint-%d/' % appoint

        savepath = 'Records-Result-OpenSmile/%s-Appoint-%d/' % (conf, appoint)
        if os.path.exists('Records-Result-OpenSmile/%s-Appoint-%d\\' % (conf, appoint)): continue

        os.makedirs('Records-Result-OpenSmile/%s-Appoint-%d/Decode' % (conf, appoint))
        os.makedirs('Records-Result-OpenSmile/%s-Appoint-%d/Logits' % (conf, appoint))
        os.makedirs('Records-Result-OpenSmile/%s-Appoint-%d/SoftMax' % (conf, appoint))

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(loadpath='E:/OpenSmileResults/%s-Npy/Appoint-%d/' % (conf, appoint))
        trainScription, testScription = IEMOCAP_Transcription_Loader_Npy_New(
            loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU-Npy-Improve/Appoint-%d/' % appoint)
        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                 trainSeq=trainSeq, testData=testData,
                                                 testLabel=testScription, testSeq=testSeq)
        # exit()
        for episode in range(100):
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=trainData[0].shape[1], numClass=5, learningRate=1e-3, batchSize=64,
                                       startFlag=False)

                classifier.Load(netpath + '%04d-Network' % episode)
                matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                       testLabel=testLabel,
                                                                                       testSeq=testSeq)
                print(matrixDecode)
                print(matrixLogits)
                print(matrixSoftMax)
                file = open('Records-Result-OpenSmile/%s-Appoint-%d/Decode/%04d.csv' % (conf, appoint, episode), 'w')
                for indexX in range(4):
                    for indexY in range(4):
                        if indexY != 0: file.write(',')
                        file.write(str(matrixDecode[indexX][indexY]))
                    file.write('\n')
                file.close()
                file = open('Records-Result-OpenSmile/%s-Appoint-%d/Logits/%04d.csv' % (conf, appoint, episode), 'w')
                for indexX in range(4):
                    for indexY in range(4):
                        if indexY != 0: file.write(',')
                        file.write(str(matrixLogits[indexX][indexY]))
                    file.write('\n')
                file.close()
                file = open('Records-Result-OpenSmile/%s-Appoint-%d/SoftMax/%04d.csv' % (conf, appoint, episode), 'w')
                for indexX in range(4):
                    for indexY in range(4):
                        if indexY != 0: file.write(',')
                        file.write(str(matrixSoftMax[indexX][indexY]))
                    file.write('\n')
                file.close()
                # exit()
