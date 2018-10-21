import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
from CTC_Project_Again.Model.CRF_BLSTM_Test import CRF_BLSTM
import os

if __name__ == '__main__':
    bands = 60

    for appoint in range(6):
        loadpath = 'D:/ProjectData/Project-CTC-Data/Records-CRF-BLSTM-Improve/Bands-' + str(bands) + '-' + str(
            appoint) + '/'
        savepath = 'D:/ProjectData/Project-CTC-Data/Records-Result-CRF-BLSTM-Improve/Bands-' + str(bands) + '-' + str(
            appoint) + '/'
        if os.path.exists(savepath): continue
        os.makedirs(savepath)

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))
        trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
            loadpath='D:/ProjectData/Project-CTC-Data/CTC-SeqLabel-Class5-Improve/Bands-' + str(bands) + '-' + str(
                appoint) + '/')

        for episode in range(100):
            if os.path.exists(savepath + '%04d.csv' % episode):
                continue
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CRF_BLSTM(trainData=trainData, trainLabel=trainSeqLabel, trainSeqLength=trainSeqLabel,
                                       featureShape=bands, numClass=4, startFlag=False, batchSize=64)
                print(loadpath + '%04d-Network' % episode)
                classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
                matrix, correctNumber, totalNumber = classifier.Test_Decode_GroundLabel(testData=testData,
                                                                                        testSeqLabel=testSeqLabel,
                                                                                        testGroundLabel=testLabel,
                                                                                        testSeq=testSeq)

                file = open(savepath + '%04d.csv' % episode, 'w')
                for indexX in range(len(matrix)):
                    for indexY in range(len(matrix[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(matrix[indexX][indexY]))
                    file.write('\n')
                file.close()
        # exit()
