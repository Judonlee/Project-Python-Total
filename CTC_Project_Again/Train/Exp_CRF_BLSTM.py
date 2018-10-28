import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Model.CRF_BLSTM_Test import CRF_BLSTM
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    part = 'WA'
    for bands in [80]:
        for appoint in [7]:
            savepath = 'Records-CRF-BLSTM-Improve-Choosed-' + part + '/Bands-' + str(bands) + '-' + str(appoint) + '/'

            if os.path.exists(savepath): continue
            os.makedirs(savepath)

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
                IEMOCAP_Loader_Npy(
                    loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-%d-%d/' % (bands, appoint))
            trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
                loadpath='D:/ProjectData/Project-CTC-Data/CTC-SeqLabel-Class5-Improve-Choosed-' + part + '/Bands-%d-%d/' % (
                    bands, appoint))

            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainSeqLabel, trainSeq=trainSeq,
                                                     testData=testData, testLabel=testSeqLabel, testSeq=testSeq)

            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CRF_BLSTM(trainData=dataClass.trainData, trainLabel=dataClass.trainLabel,
                                       trainSeqLength=dataClass.trainSeq, featureShape=bands, numClass=4,
                                       learningRate=1e-3, batchSize=64)
                print(classifier.information)

                for episode in range(100):
                    loss = classifier.Train()
                    print('\rEpisode %04d : Loss = %f' % (episode, loss))
                    # classifier.Test_Decode(testData=trainData, testLabel=trainSeqLabel, testSeq=trainSeq)
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)
                # exit()
            # exit()
