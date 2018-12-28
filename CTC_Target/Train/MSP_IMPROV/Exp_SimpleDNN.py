from CTC_Target.Loader.MSP_Loader import Loader
from CTC_Target.Model.SimpleDNN import DNN
import numpy
import os
import tensorflow

if __name__ == '__main__':
    usedPart = 'GeMAPSv01a'
    for appointSession in range(1, 7):
        for gender in ['F', 'M']:
            savepath = 'E:/CTC_Target_MSP/DNN-DropOut/%s/Session%d-%s/' % (usedPart, appointSession, gender)
            if os.path.exists(savepath): continue
            os.makedirs(savepath)

            trainData, trainLabel, testData, testLabel = Loader(
                loadpath='E:/CTC_Target_MSP/Feature/%s-Npy/' % usedPart, appointSession=appointSession,
                appointGender=gender)
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = DNN(trainData=trainData, trainLabel=trainLabel, featureShape=numpy.shape(trainData)[1],
                                 numClass=numpy.shape(trainLabel)[1], hiddenNodules=256)

                for episode in range(100):
                    print('\nBatch %d : Total Loss = %f' % (episode, classifier.Train()))
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)
