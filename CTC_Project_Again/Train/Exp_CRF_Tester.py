import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_SeqLabelLoader
import numpy
from CTC_Project_Again.Model.CRF_NN_Test import CRF_Test
import os


def FrameWiseLabelTransformation(labels, seqLen):
    result = []
    for index in range(len(labels)):
        current = numpy.ones(seqLen[index]) * (numpy.argmax(numpy.array(labels[index])) + 1)
        result.append(current)
    return result


if __name__ == '__main__':
    for bands in [30]:
        for appoint in range(10):
            loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Records-CRF-NN\\' + str(bands) + '-' + str(appoint) + '\\'
            savepathMatrix = 'D:\\ProjectData\\Project-CTC-Data\\Records-CRF-NN-Result-Matrix\\' + str(
                bands) + '-' + str(appoint) + '\\'
            savepathCounter = 'D:\\ProjectData\\Project-CTC-Data\\Records-CRF-NN-Result\\' + str(
                bands) + '-' + str(appoint) + '\\'
            if os.path.exists(savepathMatrix): continue
            os.makedirs(savepathMatrix)
            os.makedirs(savepathCounter)

            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
                loadpath='D:/ProjectData/Records-BLSTM-CTC-Normalized/Logits-Class5/' +
                         str(bands) + '-' + str(appoint) + '/')

            for episode in range(100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CRF_Test(trainData=trainData, trainLabel=trainSeqLabel, trainSeqLength=trainSeqLabel,
                                          featureShape=bands, numClass=4, startFlag=False)
                    classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
                    matrix, correctNumber, totalNumber = classifier.Test_Decode_GroundLabel(testData=trainData,
                                                                                            testLabel=trainSeqLabel,
                                                                                            testGroundLabel=trainLabel,
                                                                                            testSeq=trainSeq)
                file = open(savepathMatrix + '%04d.csv' % episode, 'w')
                for indexX in range(len(matrix)):
                    for indexY in range(len(matrix[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(matrix[indexX][indexY]))
                    file.write('\n')
                file.close()

                file = open(savepathCounter + '%04d.csv' % episode, 'w')
                file.write(str(correctNumber) + ',' + str(totalNumber))
                file.close()
