from CTC_Target.Loader.MSP_Loader import Loader
from CTC_Target.Model.SimpleDNN_WithDropOut import DNN_WithDropOut
import numpy
import os
import tensorflow

if __name__ == '__main__':
    usedPart = 'GeMAPSv01a'
    for appointSession in range(1, 7):
        for gender in ['F', 'M']:
            netpath = 'E:/CTC_Target_MSP/DNN-DropOut/%s/Session%d-%s/' % (usedPart, appointSession, gender)
            savepath = 'E:/CTC_Target_MSP/DNN-DropOut/%s-Result/Session%d-%s/' % (usedPart, appointSession, gender)
            if os.path.exists(savepath): continue
            os.makedirs(savepath)

            trainData, trainLabel, testData, testLabel = Loader(
                loadpath='E:/CTC_Target_MSP/Feature/%s-Npy/' % usedPart, appointSession=appointSession,
                appointGender=gender)
            for episode in range(100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = DNN_WithDropOut(trainData=trainData, trainLabel=trainLabel, featureShape=numpy.shape(trainData)[1],
                                     numClass=numpy.shape(trainLabel)[1], hiddenNodules=256, startFlag=False)
                    classifier.Load(loadpath=netpath + '%04d-Network' % episode)
                    matrix = classifier.Test(testData=testData, testLabel=testLabel)
                    with open(savepath + '%04d.csv' % episode, 'w') as file:
                        for indexA in range(numpy.shape(matrix)[0]):
                            for indexB in range(numpy.shape(matrix)[1]):
                                if indexB != 0: file.write(',')
                                file.write(str(matrix[indexA][indexB]))
                            file.write('\n')
                    # exit()
