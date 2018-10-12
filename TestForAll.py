import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Model.CRF_BLSTM_Test import CRF_BLSTM
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bands = 40
    appoint = 2
    startPosition = 59
    savepath = 'Records-CRF-BLSTM-Class4-Tanh/Bands-' + str(bands) + '-' + str(appoint) + '/'

    graph = tensorflow.Graph()
    with graph.as_default():
        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(
                loadpath='/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/Bobs/Npy-TotalWrapper/Bands-%d-%d/' % (
                    bands, appoint))
        trainSeqLabel, testSeqLabel = IEMOCAP_SeqLabelLoader(
            loadpath='/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/Bobs/Npy-SeqLabel/Bands-%d-%d/' % (bands, appoint))

        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainSeqLabel, trainSeq=trainSeq,
                                                 testData=testData, testLabel=testSeqLabel, testSeq=testSeq)
        # exit()
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CRF_BLSTM(trainData=dataClass.trainData, trainLabel=dataClass.trainLabel,
                                   trainSeqLength=dataClass.trainSeq, featureShape=bands, numClass=4,
                                   learningRate=1e-3, batchSize=64)
            print(classifier.information)
            for episode in range(startPosition + 1, 100):
                loss = classifier.Train()
                print('\rEpisode %04d : Loss = %f' % (episode, loss))
                # classifier.Test_Decode(testData=trainData, testLabel=trainSeqLabel, testSeq=trainSeq)
                classifier.Save(savepath=savepath + '%04d-Network' % episode)
