from CTC_Project.Loader.IEMOCAP_CTC_Loader import IEMOCAP_CTC_Loader
from CTC_Project.Module.CTC_Reconfiguration import CTC
import tensorflow
import numpy
from time import strftime

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = IEMOCAP_CTC_Loader(bands=30, appoint=0)
    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq, featureShape=30,
                         numClass=6, batchSize=128)
        print(classifier.information)
        for episode in range(100):
            loss = classifier.Train()
            print('\rEpisode : ' + str(episode) + '\tLoss : ' + str(loss) + strftime("%Y/%m/%d %H:%M:%S"))
        # classifier.PredictOutput(data=trainData, sequence=trainSeq)
    # 还差保存参数以及计算
    # 目前的计划为使用全部的数据训练出较好的结果之后将数据保存，再进行后续操作
