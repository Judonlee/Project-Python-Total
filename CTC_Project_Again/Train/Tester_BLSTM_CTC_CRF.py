import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
from CTC_Project_Again.Model.CTC_CRF_Reuse_BLSTM_NotTrain import CTC_CRF_Reuse
import os

if __name__ == '__main__':
    bands = 30

    for appoint in range(8, 10):
        loadpath = 'D:/ProjectData/Records-BLSTM-CTC-CRF-Improve-WA/Bands-%d-%d/' % (bands, appoint)
        savepath = 'D:/ProjectData/Records-Result-BLSTM-CTC-CRF-Improve-WA/Bands-%d-%d/' % (bands, appoint)
        if os.path.exists(savepath): continue
        os.makedirs(savepath)

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(
                loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))
        trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
            loadpath='D:/ProjectData/Project-CTC-Data/CTC-SeqLabel-Class5-Improve-Choosed-UA/Bands-' + str(
                bands) + '-' + str(appoint) + '/')

        for episode in range(100):
            if os.path.exists(savepath + '%04d.csv' % episode):
                continue
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_CRF_Reuse(trainData=trainData, trainLabel=trainSeqLabel, trainSeqLength=trainSeqLabel,
                                           featureShape=bands, numClass=5, startFlag=False, batchSize=64)
                print(loadpath + '%04d-Network' % episode)
                classifier.Load(loadpath=loadpath + '%04d-Network' % episode)
                # exit()
                matrix = classifier.Test_CRF(testData=testData, testLabel=testLabel, testSeq=testSeq)

                file = open(savepath + '%04d.csv' % episode, 'w')
                for indexX in range(len(matrix)):
                    for indexY in range(len(matrix[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(matrix[indexX][indexY]))
                    file.write('\n')
                file.close()
        # exit()
