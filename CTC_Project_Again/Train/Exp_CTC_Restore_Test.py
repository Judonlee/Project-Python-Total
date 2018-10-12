from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM_RestoreTest import CTC_BLSTM
import tensorflow
from CTC_Project_Again.Model.HugeNetTest import BLSTM_CTC_BLSTM_CRF

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    bands = 30
    appoint = 0
    trace = 99
    netPath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/%04d-Network' \
              % (bands, appoint, trace)

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))

    classifier = BLSTM_CTC_BLSTM_CRF(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                     featureShape=bands, numClass=5, batchSize=64, startFlag=False)
    classifier.session.run(tensorflow.global_variables_initializer())
    classifier.Load(loadpath=netPath)
    print(classifier.information)
    # classifier.Train_CRF()
    print('\n\n')
    classifier.Test_LogitsPooling_InSide(testData=testData, testLabel=testLabel, testSeq=testSeq)
