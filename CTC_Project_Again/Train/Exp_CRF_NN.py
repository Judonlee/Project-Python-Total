import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_SeqLabelLoader
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Model.CRF_NN_Test import CRF_Test
import numpy
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            savepath = 'Records-CRF-NN/' + str(bands) + '-' + str(appoint) + '/'
            if os.path.exists(savepath): continue
            os.makedirs(savepath)
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                               appoint=appoint)
            trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
                loadpath='D:/ProjectData/Records-BLSTM-CTC-Normalized/Logits-Class5/' +
                         str(bands) + '-' + str(appoint) + '/')

            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainSeqLabel, trainSeq=trainSeq,
                                                     testData=testData, testLabel=testSeqLabel, testSeq=testSeq)

            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CRF_Test(trainData=dataClass.trainData, trainLabel=dataClass.trainLabel,
                                      trainSeqLength=dataClass.trainSeq, featureShape=bands, numClass=4,
                                      learningRate=1e-3, batchSize=64)
                print(classifier.information)
                for episode in range(100):
                    print('\nEpoch', episode)
                    print('\rTotalLoss : %f' % classifier.Train())
                    # classifier.Test_Decode(testData=trainData, testLabel=trainSeqLabel, testSeq=trainSeq)
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)

            # exit()
