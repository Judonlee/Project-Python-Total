from CTC_Project_Again.Loader.IEMOCAP_Loader_New import Loader, TranscriptionLoader
from CTC_Project_Again.ModelNew.CTC_Single_BLSTM_BatchNormalization import CTC_BLSTM
import tensorflow
import os

if __name__ == '__main__':
    for bands in [30, 40, 60, 80, 100, 120]:
        for session in range(1, 6):
            loadpath = 'D:/ProjectData/Project-CTC-Data/Csv-Npy/Bands%d/' % bands
            savepath = 'Result-CTC-SingleBLSTM/Bands-%d-Session-%d/' % (bands, session)
            if os.path.exists(savepath): continue
            os.makedirs(savepath)
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(loadpath=loadpath, session=session)
            trainScription, testScription = TranscriptionLoader(loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Tran-CMU-Npy/',
                                                                session=session)
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                       featureShape=bands, numClass=5, graphRevealFlag=False, batchSize=32)
                print(classifier.information)
                for epoch in range(100):
                    print('\rEpoch %d: Total Loss = %f' % (epoch, classifier.Train()))
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                           testLabel=testLabel,
                                                                                           testSeq=testSeq)
                    classifier.Save(savepath=savepath + '%04d-Network' % epoch)
                    exit()
