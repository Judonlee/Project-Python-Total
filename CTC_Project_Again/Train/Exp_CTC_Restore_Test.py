from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
import tensorflow
from CTC_Project_Again.Model.CTC_CRF_Reuse_DoubleLoss import CTC_CRF_Reuse
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    appoint = 0
    trace = 99
    netPath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/%04d-Network' \
              % (bands, appoint, trace)
    savepath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CRF-Reuse-Restart/Bands-%d-%d/' % (bands, appoint)

    os.makedirs(savepath)

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))

    classifier = CTC_CRF_Reuse(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                               featureShape=bands, numClass=5, batchSize=64, startFlag=False)
    classifier.session.run(tensorflow.global_variables_initializer())
    classifier.LoadPart(loadpath=netPath)
    print(classifier.information)
    # exit()
    for episode in range(100):
        if episode < 10:
            print('\n')
            classifier.Test_LogitsPooling(testData=testData, testLabel=testLabel, testSeq=testSeq)
            print('\n')
        print('\nEpisode %d Total Loss = %f' % (episode, classifier.CRF_Train()))

        classifier.Save(savepath=savepath + '%04d-Network' % episode)
