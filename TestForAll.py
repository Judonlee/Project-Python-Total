from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.HugeNetTest import BLSTM_CTC_BLSTM_CRF
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bands = 30
    appoint = 0
    startPosition = 90
    savepath = 'D:/GitHub/CTC_Project_Again/Train/Records-HugeNetwork/Bands-' + str(bands) + '-' + str(appoint) + '/'

    graph = tensorflow.Graph()
    with graph.as_default():
        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
            loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                 trainSeq=trainSeq, testData=testData,
                                                 testLabel=testScription, testSeq=testSeq)
        classifier = BLSTM_CTC_BLSTM_CRF(trainData=trainData, trainLabel=trainScription,
                                         trainSeqLength=trainSeq, featureShape=bands, numClass=5,
                                         learningRate=1e-3, batchSize=64, startFlag=False)
        print(classifier.information)
        classifier.Load(savepath + '%04d-Network' % startPosition)
        # exit()

        for epoch in range(startPosition + 1, 100):
            print('\rEpoch %d: Total Loss = %f' % (epoch, classifier.Train_CRF()))
            classifier.Save(savepath=savepath + '%04d-Network' % epoch)
