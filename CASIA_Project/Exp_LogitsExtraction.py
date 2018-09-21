from CTC_Project.Loader.IEMOCAP_CTC_Loader import IEMOCAP_CTC_Loader
from CTC_Project.Module.CTC_Reconfiguration import CTC
import tensorflow

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = IEMOCAP_CTC_Loader(bands=30, appoint=-1)
    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq, featureShape=30,
                         numClass=6, startFlag=False)
        classifier.Load(loadpath=r'F:\CTC-NeuralNetwork\Bands30\NeuralParameter')
        classifier.LogitsCalculation(testData=trainData, testSeq=trainSeq)
