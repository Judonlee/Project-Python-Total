import tensorflow
from CASIA_Project.Module.BLSTM_20180919 import BLSTM
from CASIA_Project.Loader.NpyLoader import CASIA_Loader
from CASIA_Project.Engine.Engine_TrainDevelopTest import Engine_TrainDevelopTest
from CASIA_Project.Utils.DataClass import DataClass

if __name__ == '__main__':
    confList = ['avec2011', 'avec2013', 'ComParE', 'eGeMAPS', 'emobase2010', 'IS09', 'IS10', 'IS11', 'IS12', 'IS13']
    for time in ['2S', '4S', '6S', '8S', '10S', '12S']:
        for conf in confList:
            loadpath = 'F:\\AVEC-Final\\TimeThreshold' + time + '\\Features-Npy\\' + conf + '\\'
            trainData, trainLabel, developData, developLabel, testData, testLabel = CASIA_Loader(path=loadpath)
            dataClass = DataClass(trainData=trainData, trainLabel=trainLabel, developData=developData,
                                  developLabel=developLabel, testData=testData, testLabel=testLabel)

            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = BLSTM(trainData=trainData, trainLabel=trainLabel, featureShape=len(trainData[0][0]),
                                   learningRate=1e-6)
                Engine_TrainDevelopTest(dataClass=dataClass, classifier=classifier,
                                        savePath='F:\\AVEC-Final\\NetworkChangedAgain-TanhLastNot\\' + time + '-' + conf + '\\')
