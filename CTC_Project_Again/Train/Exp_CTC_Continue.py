from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM_LR_Changed import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bands = 30
    appoint = 0
    savepath = 'Records-CTC-Class5-Again-LR1E-4/Bands-' + str(bands) + '-' + str(appoint) + '/'

    graph = tensorflow.Graph()
    with graph.as_default():
        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = IEMOCAP_Loader_Npy(
            loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                 trainSeq=trainSeq, testData=testData,
                                                 testLabel=testScription, testSeq=testSeq)
        classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                               featureShape=bands, numClass=5, learningRate=1e-3, batchSize=64, startFlag=False)
        classifier.Load(loadpath=savepath + '%04d-Network' % 56)
        print(classifier.information)
        # exit()
        for epoch in range(100):
            if epoch <= 56:
                classifier.learningRate *= 0.98
                print('Epoch %d : %f' % (epoch, classifier.learningRate))
                continue
            # continue
            print('\rEpoch %d: Total Loss = %f' % (epoch, classifier.Train()))
            classifier.Save(savepath=savepath + '%04d-Network' % epoch)
