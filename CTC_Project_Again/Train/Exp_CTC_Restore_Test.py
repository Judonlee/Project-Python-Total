from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
import tensorflow
from CTC_Project_Again.Model.CTC_CRF_Reuse_BLSTM_NotTrain import CTC_CRF_Reuse
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    appoint = 1
    savepath = 'D:/Result/Bands-%d-%d-WA/' % (bands, appoint)

    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))

    for episode in range(100):
        netPath = 'D:/ProjectData/Project-CTC-Data/Records-BLSTM-CTC-CRF/Records-BLSTM-CTC-CRF-WA/Bands-%d-%d/%04d-Network' \
                  % (bands, appoint, episode)
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_CRF_Reuse(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, batchSize=64, startFlag=False)
            classifier.Load(loadpath=netPath)
            # print(classifier.information)
            # exit()
            # for episode in range(100):
            # if episode < 10:
            #     print('\n')
            #     classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
            #     print('\n')
            # print('\nEpisode %d Total Loss = %f' % (episode, classifier.CRF_Train()))
            print('\n\nEpisode', episode)
            matrix = classifier.Test_CRF(testData=testData, testLabel=testLabel, testSeq=testSeq)
            file = open(savepath + '%04d-Result.csv' % episode, 'w')
            for indexX in range(len(matrix)):
                for indexY in range(len(matrix[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(matrix[indexX][indexY]))
                file.write('\n')
            file.close()

            # classifier.Save(savepath=savepath + '%04d-Network' % episode)
