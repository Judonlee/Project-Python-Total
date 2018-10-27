from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    savepath = 'D:/ProjectData/Visualization/'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in [5]:
            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))
            netpath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New-Choosed/Bands-%d-%d/UA' % (bands, appoint)
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1,
                                       startFlag=False, batchSize=64)
                classifier.Load(loadpath=netpath)
                # matrixA, matrixB, matrixC = classifier.Test_AllMethods(testData=testData, testLabel=testLabel,
                #                                                        testSeq=testSeq)
                result = classifier.Visualization(testData=trainData, testSeq=trainSeq)
                print(numpy.shape(result))
                for index in range(len(result)):
                    print(index)
                    if not os.path.exists(savepath + str(numpy.argmax(numpy.array(trainLabel[index])))):
                        os.makedirs(savepath + str(numpy.argmax(numpy.array(trainLabel[index]))))
                    file = open(
                        savepath + str(numpy.argmax(numpy.array(trainLabel[index]))) + '/' + str(index) + '.csv', 'w')
                    for indexX in range(trainSeq[index]):
                        for indexY in range(len(result[index][indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(result[index][indexX][indexY]))
                        file.write('\n')
                    file.close()
                exit()
