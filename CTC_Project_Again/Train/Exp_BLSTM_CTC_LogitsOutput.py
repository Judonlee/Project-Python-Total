from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
from CTC_Project_Again.Model.CTC_BLSTM_NN import CTC_BLSTM_NN
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bands = 30
    appoint = 9
    episode = 98
    savepath = 'D:\\ProjectData\\Records-BLSTM-CTC-Normalized\\Logits-Class5\\' + str(bands) + '-' + str(appoint) + '\\'
    if not os.path.exists(savepath): os.makedirs(savepath)

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
        IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                       appoint=appoint)
    trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
        loadpath='D:/ProjectData/Project-CTC-Data/Transcription-SingleNumber-Class5/', appoint=appoint)
    dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                             trainSeq=trainSeq, testData=testData,
                                             testLabel=testTranscription, testSeq=testSeq)
    # exit()
    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                               featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1,
                               startFlag=False)
        classifier.Load('D:\\ProjectData\\Records-BLSTM-CTC-Normalized\\Bands-' + str(bands) + '-'
                        + str(appoint) + '\\%04d-Network' % episode)
        logits = classifier.LogitsOutput(testData=trainData, testSeq=trainSeq)

        sequenceLabel = []
        for indexX in range(len(logits)):
            currentLabel = []
            for indexY in range(len(logits[indexX])):
                currentLabel.append(numpy.argmax(numpy.array(logits[indexX][indexY][0:4])))
            sequenceLabel.append(currentLabel)

        numpy.save(savepath + 'TrainSeqLabel.npy', sequenceLabel)

        logits = classifier.LogitsOutput(testData=testData, testSeq=testSeq)

        sequenceLabel = []
        for indexX in range(len(logits)):
            currentLabel = []
            for indexY in range(len(logits[indexX])):
                currentLabel.append(numpy.argmax(numpy.array(logits[indexX][indexY][0:4])))
            sequenceLabel.append(currentLabel)

        numpy.save(savepath + 'TestSeqLabel.npy', sequenceLabel)
    # exit()
