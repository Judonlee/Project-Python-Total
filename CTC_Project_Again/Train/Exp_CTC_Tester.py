from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC import CTC
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            savepath = 'F:/Project-CTC-Data/Records-CTC-RNN2nd/Bands-' + str(bands) + '-' + str(appoint) + '/'
            graph = tensorflow.Graph()
            with graph.as_default():
                trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                    IEMOCAP_Loader(loadpath='F:/Project-CTC-Data/Npy/Bands' + str(bands) + '/', appoint=appoint)
                trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
                    loadpath='F:/Project-CTC-Data/Transcription-SingleNumber/', appoint=appoint)
                dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                         trainSeq=trainSeq, testData=testData,
                                                         testLabel=testTranscription, testSeq=testSeq)
                classifier = CTC(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                 featureShape=bands, numClass=6, learningRate=5e-5, rnnLayers=1, startFlag=False)
                classifier.Load(r'F:\Project-CTC-Data\Records-CTC\Bands-30-0\0040-Network')
                print(classifier.information)
                # classifier.Test(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
                logits, dense = classifier.session.run(
                    fetches=[classifier.parameters['Logits_Reshape'], classifier.decodeDense],
                    feed_dict={classifier.dataInput: trainData[0:1],
                               classifier.seqLenInput: trainSeq[0:1]})
                print(numpy.shape(logits))
                print(dense)
                for index in range(numpy.shape(logits)[1]):
                    print(logits[0][index])
                exit()
