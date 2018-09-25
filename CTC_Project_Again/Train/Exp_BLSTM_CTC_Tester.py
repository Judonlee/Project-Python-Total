from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            savepath = 'F:/Project-CTC-Data/Records-BLSTM-CTC/Bands-' + str(bands) + '-' + str(appoint) + '/'
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                IEMOCAP_Loader(loadpath='F:/Project-CTC-Data/Npy/Bands' + str(bands) + '/', appoint=appoint)
            trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
                loadpath='F:/Project-CTC-Data/Transcription-SingleNumber-Class5/', appoint=appoint)
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=testTranscription, testSeq=testSeq)
            for episode in range(8):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                           featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1,
                                           startFlag=False)
                    # print(classifier.information)
                    print('F:\\Project-CTC-Data\\Records-BLSTM-CTC\\Bands-30-0\\%04d-Network' % episode)
                    classifier.Load('F:\\Project-CTC-Data\\Records-BLSTM-CTC\\Bands-30-0\\%04d-Network' % episode)
                    matrix = classifier.Test_Decode(testData=testData, testLabel=testLabel, testSeq=testSeq)

                    print()

            exit()
