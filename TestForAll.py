from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_SeqLabelLoader
import numpy
import os

if __name__ == '__main__':
    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            savepath = 'Records-CRF-NN/' + str(bands) + '-' + str(appoint) + '/'
            if os.path.exists(savepath): continue
            # os.makedirs(savepath)
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
                loadpath='D:/ProjectData/Records-BLSTM-CTC-Normalized/Logits-Class5/' +
                         str(bands) + '-' + str(appoint) + '/')
            print(len(trainData), len(testData), len(trainSeqLabel), len(testSeqLabel))
            for index in range(len(trainData)):
                print(numpy.shape(trainData[index]), numpy.shape(trainSeqLabel[index]))
            exit()
