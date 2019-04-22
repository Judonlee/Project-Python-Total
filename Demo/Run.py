from Demo.Loader import Loader
from Demo.Model import NetworkStructure
import os
import tensorflow

if __name__ == '__main__':
    session = 1
    gender = 'F'

    trainData, trainLabel, valData, valLabel, testData, testLabel = Loader(
        appointSession=session, appointGender=gender, datapath='D:/ProjectData/data/data/',
        labelpath='D:/ProjectData/data/one_hot_labels/')
    # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(valData), numpy.shape(valLabel),
    #       numpy.shape(testData), numpy.shape(testLabel))

    savepath = 'Matrixs/Session%d-%s/' % (session, gender)
    os.makedirs(savepath)

    classifier = NetworkStructure(trainData=trainData, trainLabel=trainLabel, developData=valData,
                                  developLabel=valLabel, batchSize=32)
    tensorflow.summary.FileWriter('logs/', classifier.session.graph)
    for episode in range(100):
        print('\nEpisode %d Train Total Loss = %f' % (episode, classifier.Train()))
        classifier.Test(logname=savepath + '%04d.csv' % episode, testData=testData, testLabel=testLabel)
