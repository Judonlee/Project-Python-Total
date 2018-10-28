from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
import tensorflow
from CTC_Project_Again.Model.BLSTM_CTC_CRF_Attention import BLSTM_CTC_CRF_Attention
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            netPath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New-Choosed/Bands-%d-%d/WA' % (
                bands, appoint)
            savepath = 'Records-BLSTM-CTC-CRF-Attention/Bands-%d-%d/' % (bands, appoint)

            if os.path.exists(savepath): continue
            os.makedirs(savepath)

            graph = tensorflow.Graph()
            with graph.as_default():
                trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
                    IEMOCAP_Loader_Npy(
                        loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (
                            bands, appoint))

                classifier = BLSTM_CTC_CRF_Attention(trainData=trainData, trainLabel=trainScription,
                                                     trainSeqLength=trainSeq, featureShape=bands, numClass=5,
                                                     batchSize=64, startFlag=True)
                classifier.LoadPart(loadpath=netPath)
                print(classifier.information)
                for episode in range(100):
                    if episode < 10:
                        classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
                    classifier.CRF_Train()
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)
