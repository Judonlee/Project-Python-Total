from CTC_Project_Perhaps_Failed.Loader.IEMOCAP_CTC_Loader import IEMOCAP_CTC_Loader
from CTC_Project_Perhaps_Failed.Module.CTC_Reconfiguration import CTC
import tensorflow
import numpy
import os
from time import strftime

if __name__ == '__main__':
    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = IEMOCAP_CTC_Loader(bands=bands,
                                                                                               appoint=appoint)
            savepath = 'F:\\CTC-NeuralNetwork-Seperate\\Bands' + str(bands) + '\\' + str(appoint) + '\\'
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                 featureShape=bands,
                                 numClass=6, batchSize=128, learningRate=1e-5)
                print(classifier.information)
                for episode in range(100):
                    loss = classifier.Train()
                    print('\rEpisode : ' + str(episode) + '\tLoss : ' + str(loss) + strftime("%Y/%m/%d %H:%M:%S"))
                    classifier.LogitsCalculation(testData=trainData, testSeq=trainSeq)
                #os.makedirs(savepath)
                #classifier.Save(savepath=savepath + 'NeuralParameter')
                # classifier.PredictOutput(data=trainData, sequence=trainSeq)
            # 还差保存参数以及计算
            # 目前的计划为使用全部的数据训练出较好的结果之后将数据保存，再进行后续操作
