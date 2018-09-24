from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader
from CTC_Project_Again.Model.LSTM_FinalPooling import LSTM_FinalPooling
from CTC_Project_Again.Model.LSTM_AveragePooling import LSTM_AveragePooling
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Engine.TrainTestEngine import TrainTestEngine
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(1, 10, 2):
            savepath = 'F:/Project-CTC-Data/Records-AveragePooing/Bands-' + str(bands) + '-' + str(appoint) + '/'
            graph = tensorflow.Graph()
            with graph.as_default():
                trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                    IEMOCAP_Loader(loadpath='F:/Project-CTC-Data/Npy/Bands' + str(bands) + '/', appoint=appoint)
                dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                                                         testData=testData, testLabel=testLabel, testSeq=testSeq)
                classifier = LSTM_AveragePooling(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                                 featureShape=bands, numClass=4, rnnLayers=1)
                print(classifier.information)
                TrainTestEngine(dataClass=dataClass, classifier=classifier, totalEpoch=100, savepath=savepath)
