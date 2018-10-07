from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bands = 30
    episode = 99
    for appoint in range(4):
        savepath = 'D:/ProjectData/Project-CTC-Data/CTC-SeqLabel-Class5/Bands-%d-%d/' % (bands, appoint)
        netpath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/' % (bands, appoint)

        if not os.path.exists(savepath): os.makedirs(savepath)

        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
            IEMOCAP_Loader_Npy(loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/'
                                        % (bands, appoint))
        dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                 trainSeq=trainSeq, testData=testData,
                                                 testLabel=testScription, testSeq=testSeq)
        # exit()
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=5, learningRate=5e-5, rnnLayers=1, startFlag=False,
                                   batchSize=64)
            classifier.Load(netpath + '%04d-Network' % episode)
            # exit()
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
